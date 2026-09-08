import argparse
import ast
import json
import re
from collections import deque
from pathlib import Path
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


QUERY_STRUCTS = ['1p', '2p', '2i', '3i', '2u']
SYSTEM_PROMPT = (
    "You are an expert in deductive reasoning over the supplied knowledge "
    "graph facts. Use only the provided facts. Output only the answer "
    "entities separated by English commas, without brackets, labels, or "
    "explanations. If no entity satisfies the query, output only 'None'."
)
SYSTEM_PROMPT_COT = (
    "You are an expert in deductive reasoning over the supplied knowledge "
    "graph facts. Use only the provided facts and solve the problem step by "
    "step. Clearly derive the intermediate entity sets. After completing the "
    "reasoning, the last non-empty line must contain only the final answer "
    "entities separated by English commas, without brackets, labels, or "
    "explanations. If no entity satisfies the query, the last non-empty line "
    "must be exactly 'None'."
)
SYSTEM_PROMPT_FEWSHOT_COT = (
    "You are an expert in deductive reasoning over the supplied knowledge "
    "graph facts. Follow the reasoning style and output format demonstrated "
    "in the examples. The last non-empty line must be a valid answer object "
    "in exactly this form: Answer: {\"answer\": [\"entity1\", \"entity2\"]}. "
    "If there is no answer, use Answer: {\"answer\": []}. Do not write "
    "anything after the final answer line."
)

def local_model(model_path, attn_implementation="sdpa"):
    assert torch.cuda.is_available(), "CUDA GPU is required."

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.generation_config.use_cache = True
    return model, tokenizer
def build_chat_text(tokenizer, system_prompt, prompt):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
def build_dynamic_batches(
    lengths,
    max_batch_size,
    max_batch_tokens,
    max_new_tokens,
):
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches = []
    batch = []
    batch_max_length = 0

    for index in order:
        new_max_length = max(batch_max_length, lengths[index])
        new_batch_size = len(batch) + 1
        token_num = new_batch_size * (new_max_length + max_new_tokens)
        exceeds_limit = (
            new_batch_size > max_batch_size
            or token_num > max_batch_tokens
        )

        if batch and exceeds_limit:
            batches.append(batch)
            batch = []
            batch_max_length = 0

        batch.append(index)
        batch_max_length = max(batch_max_length, lengths[index])

    if batch:
        batches.append(batch)
    return batches

def generate_batch(
    model,
    tokenizer,
    system_prompt,
    prompts,
    max_batch_size=16,
    max_batch_tokens=16000,
    max_new_tokens=512,
    max_context_length=40960,
    desc="Generating",
):
    if not prompts:
        return []

    texts = [
        build_chat_text(tokenizer, system_prompt, prompt)
        for prompt in prompts
    ]
    token_ids = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]
    lengths = [len(input_ids) for input_ids in token_ids]
    responses = [""] * len(texts)
    valid_indices = [
        index
        for index, length in enumerate(lengths)
        if length + max_new_tokens <= max_context_length
    ]
    overflow_indices = sorted(set(range(len(texts))) - set(valid_indices))
    for index in overflow_indices:
        responses[index] = "CONTEXT_OVERFLOW"
    if overflow_indices:
        print(
            f"{desc}: skipped {len(overflow_indices)} prompts exceeding "
            f"{max_context_length} total tokens."
        )

    valid_lengths = [lengths[index] for index in valid_indices]
    relative_batches = build_dynamic_batches(
        valid_lengths,
        max_batch_size,
        max_batch_tokens,
        max_new_tokens,
    )
    batches = [
        [valid_indices[relative_index] for relative_index in batch]
        for batch in relative_batches
    ]

    pending_batches = deque(batches)

    with tqdm(
        total=len(batches),
        desc=desc,
        unit="batch",
        leave=False,
    ) as progress:
        while pending_batches:
            indices = pending_batches.popleft()
            batch_texts = [texts[index] for index in indices]
            model_inputs = None
            outputs = None
            try:
                model_inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                model_inputs = {
                    key: value.to(model.device)
                    for key, value in model_inputs.items()
                }
                input_length = model_inputs["input_ids"].shape[1]

                with torch.inference_mode():
                    outputs = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        # temperature=0.05,
                        # top_p=0.8,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                decoded = tokenizer.batch_decode(
                    outputs[:, input_length:],
                    skip_special_tokens=True,
                )
                for index, answer in zip(indices, decoded):
                    responses[index] = answer
                del outputs, model_inputs
                progress.update(1)
            except torch.OutOfMemoryError:
                if outputs is not None:
                    del outputs
                if model_inputs is not None:
                    del model_inputs
                torch.cuda.empty_cache()

                if len(indices) == 1:
                    responses[indices[0]] = "CUDA_OOM"
                    progress.update(1)
                    continue

                middle = len(indices) // 2
                pending_batches.appendleft(indices[middle:])
                pending_batches.appendleft(indices[:middle])
                progress.total += 1
                progress.refresh()
    return responses
def swap_question_placeholders(enhance_query, llm_answer):
    """将问题中的占位符替换为对应的问题答案。"""
    def replace(match):
        answer_index = int(match.group(1)) - 1
        if answer_index >= len(llm_answer):
            raise ValueError(
                f"Placeholder {match.group(0)} refers to a missing answer."
            )
        answer = llm_answer[answer_index]
        answer = "None" if answer is None else str(answer)
        return "{" + answer + "}"

    return re.sub(r"\[PP(\d+)\]", replace, enhance_query)
def parse_model_answer(raw_output):
    """Extract a normalized entity set from baseline-model output.

    Accepted high-confidence forms include a bare comma-separated answer,
    ``None``, and a complete JSON answer object. Explanations that end in an
    explicit ``Answer: ...`` line and embedded JSON answers are recovered with
    medium confidence. Unlabelled numbers inside reasoning are deliberately
    not guessed, because they can be entities or relations from KG triples.
    """
    JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
    CODE_BLOCK_PATTERN = re.compile(
        r"```(?:json|text|txt)?\s*(.*?)```",
        re.IGNORECASE | re.DOTALL,
    )
    NO_ANSWER_VALUES = {
        "none",
        "null",
        "nil",
        "n/a",
        "empty",
        "empty set",
        "[]",
        "{}",
        "∅",
        "无答案",
        "没有答案",
        "空集",
    }
    FAILURE_MARKERS = {
        "context_overflow",
        "cuda_oom",
        "skipped_previous_failure",
    }

    def format_answer(entities):
        if not entities:
            return "None"
        cleaned = []
        seen = set()
        for entity in entities:
            if isinstance(entity, bool):
                return None
            entity = str(entity).strip().strip("`\"'")
            if not entity:
                return None
            if entity.casefold() in NO_ANSWER_VALUES:
                return None
            if entity not in seen:
                seen.add(entity)
                cleaned.append(entity)

        def sort_key(entity):
            if re.fullmatch(r"-?\d+", entity):
                return 0, int(entity), entity
            return 1, entity.casefold(), entity

        return ",".join(sorted(cleaned, key=sort_key)) if cleaned else "None"

    def parse_answer_value(value):
        if isinstance(value, str):
            if value.strip().casefold() in NO_ANSWER_VALUES:
                return []
            return [value]
        if not isinstance(value, (list, tuple, set)):
            return None
        if not all(
            isinstance(item, (str, int)) and not isinstance(item, bool)
            for item in value
        ):
            return None
        return list(value)

    def parse_json_value(value):
        if isinstance(value, dict):
            for key in (
                "answers",
                "answer",
                "entities",
                "entity_ids",
                "result",
                "results",
                "output",
            ):
                if key in value:
                    return parse_answer_value(value[key])
            return None
        return parse_answer_value(value)

    def parse_plain_candidate(candidate):
        candidate = str(candidate).strip()
        candidate = candidate.replace("**", "").replace("__", "")
        candidate = candidate.strip("` \n\r\t")
        candidate = candidate.rstrip(".。!！").strip()
        if not candidate:
            return None
        if candidate.casefold() in NO_ANSWER_VALUES:
            return []

        # Support a JSON object/array when the whole candidate is JSON.
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            value = None
        if value is not None:
            parsed = parse_json_value(value)
            if parsed is not None:
                return parsed

        # Braces are often used for mathematical sets rather than JSON.
        if (
            len(candidate) >= 2
            and (candidate[0], candidate[-1])
            in {("[", "]"), ("{", "}"), ("(", ")")}
        ):
            candidate = candidate[1:-1].strip()
            if not candidate:
                return []

        # A plain answer must be one line. This prevents accidental extraction
        # of entity/relation IDs from a reasoning paragraph.
        if "\n" in candidate or "\r" in candidate:
            return None
        if re.search(
            r"\([^,()]+\s*,\s*[^,()]+\s*,\s*[^,()]+\)",
            candidate,
        ):
            return None
        if re.search(
            r"\b(?:answer|result|therefore|thus|hence|finally|because|"
            r"triples?|relations?|entities?|connected|inspect)\b|"
            r"(?:答案|结果|因此|所以|最终|三元组|关系|实体)",
            candidate,
            flags=re.IGNORECASE,
        ):
            return None
        parts = re.split(r"\s*(?:,|，|、|;|；)\s*", candidate)
        if not parts or any(not part.strip() for part in parts):
            return None
        return [part.strip().strip("`\"'") for part in parts]

    text = "" if raw_output is None else str(raw_output).strip()
    if not text or text.casefold() in FAILURE_MARKERS:
        return "failed", None

    # 1. A complete JSON answer is unambiguous.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        entities = parse_json_value(value)
        if entities is not None:
            answer = format_answer(entities)
            if answer is not None:
                return "high", answer

    # 2. The preferred answer-only protocol: ``e1,e2`` or ``None``.
    if "\n" not in text and "\r" not in text:
        entities = parse_plain_candidate(text)
        if entities is not None:
            answer = format_answer(entities)
            if answer is not None:
                return "high", answer

    # 3. For CoT responses, prefer the final explicitly labelled answer.
    answer_label_pattern = re.compile(
        r"^(?:final\s+answer|answer|final\s+result|result|"
        r"最终答案|答案|结果)\s*(?:is|are|为|是)?\s*[:：=]\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = answer_label_pattern.match(
            line.replace("**", "").replace("__", "")
        )
        if not match:
            continue
        entities = parse_plain_candidate(match.group(1))
        if entities is not None:
            answer = format_answer(entities)
            if answer is not None:
                return "medium", answer

    # 4. Recover JSON embedded in explanations or Markdown code blocks.
    candidates = CODE_BLOCK_PATTERN.findall(text)
    candidates.extend(JSON_OBJECT_PATTERN.findall(text))
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        entities = parse_json_value(value)
        if entities is not None:
            answer = format_answer(entities)
            if answer is not None:
                return "medium", answer

    # 5. Recover a concluding sentence such as ``So the answer is {...}``.
    conclusion_pattern = re.compile(
        r"(?:therefore|thus|hence|so|finally|因此|所以|最终)?\s*"
        r"(?:the\s+)?(?:final\s+)?(?:answer|result|答案|结果)\s*"
        r"(?:is|are|为|是)\s*[:：]?\s*(.+?)\s*[.。!！]?$",
        re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?。！？])\s*|\r?\n+", text)
    for sentence in reversed(sentences):
        match = conclusion_pattern.search(
            sentence.replace("**", "").replace("__", "").strip()
        )
        if not match:
            continue
        entities = parse_plain_candidate(match.group(1))
        if entities is not None:
            answer = format_answer(entities)
            if answer is not None:
                return "low", answer

    return "failed", None
def gen_local(
    dataset,
    task,
    model_name,
    base_dir,
    model_path,
    max_batch_size=16,
    max_batch_tokens=16000,
    max_new_tokens=512,
    max_context_length=40960,
    attn_implementation="sdpa",
    overwrite=False,
):
    base_dir = Path(base_dir)
    prompt_root = base_dir / "prompt"
    answer_root = base_dir / "answer"
    model_path = (
        Path(model_path)
        if model_path is not None
        else base_dir / "model" / model_name
    )
    model, tokenizer = local_model(
        model_path,
        attn_implementation,
    )
    if task == "CoT":
        system_prompt = SYSTEM_PROMPT_COT
    elif task == "FewShot_CoT":
        system_prompt = SYSTEM_PROMPT_FEWSHOT_COT
    else:
        system_prompt = SYSTEM_PROMPT

    for qtype in QUERY_STRUCTS:
        prompt_folder = prompt_root / dataset / task / qtype
        assert prompt_folder.is_dir(), f"{prompt_folder} does not exist."

        answer_folder = answer_root / dataset / model_name / task / qtype
        answer_folder.mkdir(parents=True, exist_ok=True)

        prompt_files = sorted(
            prompt_folder.glob("*.txt"),
            key=lambda path: int(path.stem),
        )
        records = []

        for prompt_file in prompt_files:
            answer_path = answer_folder / prompt_file.name
            if answer_path.exists() and not overwrite:
                continue

            prompts = ast.literal_eval(
                prompt_file.read_text(encoding="utf-8")
            )['query']
            records.append({
                "prompt_file": prompt_file,
                "queries": prompts,
                "answers": [],
                "raw_answers": [],
                "confidences": [],
                "terminal_failed": False,
            })

        if not records:
            print(f"{dataset}/{qtype}/{task}: already complete.")
            continue

        max_phase = max(len(record["queries"]) for record in records)
        for phase_index in range(max_phase):
            skipped_records = [
                record
                for record in records
                if (
                    phase_index < len(record["queries"])
                    and record["terminal_failed"]
                )
            ]
            for record in skipped_records:
                record["answers"].append("None")
                record["raw_answers"].append(
                    "SKIPPED_PREVIOUS_FAILURE"
                )
                record["confidences"].append("failed")

            active_records = [
                record
                for record in records
                if (
                    phase_index < len(record["queries"])
                    and not record["terminal_failed"]
                )
            ]
            phase_prompts = []

            for record in active_records:
                query = record["queries"][phase_index]
                if phase_index > 0:
                    query = swap_question_placeholders(
                        query,
                        record["answers"],
                    )
                phase_prompts.append(query)


            raw_answers = generate_batch(
                model,
                tokenizer,
                system_prompt,
                phase_prompts,
                max_batch_size=max_batch_size,
                max_batch_tokens=max_batch_tokens,
                max_new_tokens=max_new_tokens,
                max_context_length=max_context_length,
                desc=f"{dataset}/{qtype}/phase-{phase_index + 1}",
            )

            for record, raw_answer in zip(
                active_records,
                raw_answers,
            ):
                confidence, answer = parse_model_answer(raw_answer)
                if confidence == "failed":
                    answer = "None"
                if raw_answer in {"CONTEXT_OVERFLOW", "CUDA_OOM"}:
                    record["terminal_failed"] = True

                record["answers"].append(answer)
                record["raw_answers"].append(raw_answer)
                record["confidences"].append(confidence)

        for record in records:
            answer_path = answer_folder / record["prompt_file"].name
            result = {
                "answers": record["answers"],
                "raw_answers": record["raw_answers"],
                "confidences": record["confidences"],
                "status": (
                    "failed"
                    if "failed" in record["confidences"]
                    else "ok"
                ),
            }
            answer_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched staged inference for KG reasoning prompts."
    )
    parser.add_argument("--base-dir")
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--model-name")
    parser.add_argument("--model-path")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context-length", type=int, default=40960)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    gen_local(
        dataset=args.dataset,
        task=args.task,
        model_name=args.model_name,
        base_dir=args.base_dir,
        model_path=args.model_path,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        attn_implementation=args.attn_implementation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

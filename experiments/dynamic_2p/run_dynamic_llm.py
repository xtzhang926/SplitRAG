import argparse
import ast
import json
import pandas as pd
import re
from collections import deque
from pathlib import Path
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

QUERY_STRUCTS = ["1p", "2p", "2i", "3i", "2u"]

class StepPromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
        self.query_structs = QUERY_STRUCTS

    def parse_logical_query(self, logical_query, query_type):
        e1 = r1 = e2 = r2= e3 = r3 = None
        if query_type=="1p": (e1, (r1,)) = logical_query
        if query_type=="2p": (e1, (r1, r2)) = logical_query
        if query_type=="2i": ((e1, (r1,)), (e2, (r2,))) = logical_query
        if query_type=="3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
        if query_type=="2u": ((e1, (r1,)), (e2, (r2,)), (u,)) = logical_query

        return [e1, r1, e2, r2, e3, r3]
    def __generate_question_1p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="1p")
        entity = e1
        relation = r1
        return {1: f"Which entities are connected to {entity} by relation {relation}?"}
    def __generate_question_2p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2
        return {1: f"Which entities are connected to {entity} by relation {relation1}?",
                 2:f"Which entities are connected to entities in [PP1] by relation {relation2}?"
                }
    def __generate_question_2i(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return {1:f"Which entities are connected to {entity1} by relation {relation1}?",
                 2: f"Which entities are connected to {entity2} by relation {relation2}?",
                  3:f"Which entities exist in both sets [PP1] and [PP2]?"
                }
    def __generate_question_3i(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="3i")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {1:f"Which entities are connected to {entity1} by relation {relation1}?",
                 2:      f"Which entities are connected to {entity2} by relation {relation2}?",
                  3:     f"Which entities are connected to {entity3} by relation {relation3}?",
                   4:    f"Which entities exist in both sets [PP1], [PP2] and [PP3]?"
                }
    def __generate_question_2u(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2u")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return {1:f"Which entities are connected to {entity1} by relation {relation1}?",
                 2:f"Which entities are connected to {entity2} by relation {relation2}?",
                 3:f"What entities exist in either sets [PP1] or [PP2]?"
                }

    def __generate_question(self, logical_query, query_type):
        if query_type=="1p": return self.__generate_question_1p(logical_query)
        if query_type=="2p": return self.__generate_question_2p(logical_query)
        if query_type=="2i": return self.__generate_question_2i(logical_query)
        if query_type=="3i": return self.__generate_question_3i(logical_query)
        if query_type=="2u": return self.__generate_question_2u(logical_query)
    def generate_prompt(self, logical_query, query_type):
        question = self.__generate_question(logical_query, query_type)
        new_question = dict()
        for key, _ in question.items():
            new_question[key] = self.question_tag + question[key] + self.explain_tag
        return new_question

def parse_logical_query(logical_query, query_type):
    e1 = r1 = e2 = r2 = e3 = r3 = None
    if query_type == "1p": (e1, (r1,)) = logical_query
    if query_type == "2p": (e1, (r1, r2)) = logical_query
    if query_type == "2i": ((e1, (r1,)), (e2, (r2,))) = logical_query
    if query_type == "3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
    if query_type == "2u": ((e1, (r1,)), (e2, (r2,)), (u,)) = logical_query
    return [e1, r1, e2, r2, e3, r3]
SYSTEM_PROMPT = (
    "You are an expert in deductive reasoning over the supplied knowledge "
    "graph facts. Use only the provided facts. Output only the answer "
    "entities separated by English commas, without brackets, labels, or "
    "explanations. If no entity satisfies the query, output only 'None'."
)

PREMISE_SYSTEM = ('If any (h, r, t) triplets are provided below, they indicate '
                  'that entity h is related to entity t by relation r. Otherwise, '
                  'ignore this section.\n')


def format_triple(triple):
    head, relation, tail = triple
    return f"({head},{relation},{tail})"
def load_outgoing_triplets(kg_path):

    df = pd.read_csv(kg_path, encoding="utf-8-sig")

    outgoing_triplets = {}

    for head, relation, tail in zip(
        df["head"], df["relation"], df["tail"]
    ):
        head = int(head)
        relation = int(relation)
        tail = int(tail)

        key = (head, relation)

        outgoing_triplets.setdefault(key, set()).add(
            (head, relation, tail)
        )

    return outgoing_triplets
class DynamicPremiseGenerator:

    PHASE_SPECS = {
        "1p": [("fixed", 0, 0, "projection")],
        "2p": [
            ("fixed", 0, 0, "projection"),
            ("previous", None, 1, "projection"),
        ],
        "2i": [
            ("fixed", 0, 0, "projection"),
            ("fixed", 1, 1, "projection"),
            None,
        ],
        "3i": [
            ("fixed", 0, 0, "projection"),
            ("fixed", 1, 1, "projection"),
            ("fixed", 2, 2, "projection"),
            None,
        ],
        "2u": [
            ("fixed", 0, 0, "projection"),
            ("fixed", 1, 1, "projection"),
            None,
        ],

    }

    def __init__(self, outgoing_triplets):
        self.outgoing_triplets = outgoing_triplets

    def generate_phase_premise(
        self,
        entities,
        relations,
        previous_entities,
        phase,
        query_type,
    ):

        spec = self.PHASE_SPECS[query_type][phase - 1]
        if spec is None:
            return {
                "head_source": "none",
                "head_entities": [],
                "target_relation": None,
                "operation": "set_operation",
                "triplets": [],
            }

        head_source, entity_index, relation_index, operation = spec


        if head_source == "fixed":
            head_entities = {entities[entity_index]}
        else:

            head_entities = set(previous_entities)


        triplets = set()
        target_relation = relations[relation_index]
        for entity in head_entities:
            triplets.update(self.outgoing_triplets.get((int(entity),int(target_relation)), set()))

        return {
            "head_source": head_source,
            "head_entities": sorted(head_entities),
            "target_relation": target_relation,
            "operation": operation,
            "triplets": [list(item) for item in sorted(triplets)],
        }
def local_model(model_path, adapter=None, attn_implementation="sdpa"):
    """加载基础模型；提供adapter时将全能LoRA合并到基础模型。"""
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
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

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
def build_dynamic_batches(lengths,max_batch_size,max_batch_tokens,max_new_tokens,):

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
def parse_model_answer(raw_output):

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
        return [],[]

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

    input_stats = []
    for text, input_ids in zip(texts, token_ids):
        input_stats.append({
            "input_chars": len(text),
            "input_tokens": len(input_ids),
        })

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
    return responses, input_stats
def swap_question_placeholders(enhance_query, llm_answer):

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
def llm_answer_to_set(answer):

    if answer in {None, "None", ""}:
        return set()
    if not re.fullmatch(r"-?\d+(?:,-?\d+)*", str(answer)):
        raise ValueError(f"Invalid normalized answer: {answer}")
    return {int(value) for value in str(answer).split(",")}

def build_phase_prompt(question, premise_data):

    triplets = [tuple(item) for item in premise_data["triplets"]]
    if triplets:
        facts = ",".join(format_triple(item) for item in triplets) + ".\n"
    else:
        facts = "\n"
    return PREMISE_SYSTEM + facts + question



def save_dynamic_record(record, prompt_path, premise_path, answer_path):

    prompt_path.write_text(
        json.dumps(record["phase_prompts"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    premise_path.write_text(
        json.dumps(record["phase_premises"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    answer_result = {
        "answers": record["answers"],
        "raw_answers": record["raw_answers"],
        "confidences": record["confidences"],
        "phases": record["phase_answers"],
        "status": (
            "failed" if "failed" in record["confidences"] else "ok"
        ),
        "input_chars": record["input_chars"],

        "input_tokens": record["input_tokens"],
  
        "total_input_chars": sum(
            x for x in record["input_chars"]
            if x is not None
        ),
        "total_input_tokens": sum(
            x for x in record["input_tokens"]
            if x is not None
        ),
    }
    answer_path.write_text(
        json.dumps(answer_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
def gen_dynamic(
    model_name,
    base_dir,
    model_path,
    adapter=None,
    max_batch_size=16,
    max_batch_tokens=16000,
    max_new_tokens=256,
    max_context_length=40960,
    attn_implementation="sdpa",
    overwrite=False,
):

    base_dir = Path(base_dir)

    prompt_root = base_dir / 'dynamic' / "prompt"
    premise_root = base_dir / 'dynamic' / "premise"
    answer_root = base_dir / 'dynamic' / "answer"

    model_path = (
        Path(model_path)
        if model_path is not None
        else base_dir / "model" / model_name
    )
    adapter = None if adapter is None else Path(adapter)
    outgoing_triplets = load_outgoing_triplets(kg_path = Path(""))
    premise_generator = DynamicPremiseGenerator(outgoing_triplets)
    prompt_generator = StepPromptGenerator()
    model, tokenizer = local_model(
        model_path,
        adapter,
        attn_implementation,
    )

    for qtype in QUERY_STRUCTS:
        question_folder = base_dir / 'benchmark' / qtype / "query_abstract"

        prompt_folder = prompt_root  / model_name / qtype
        premise_folder = premise_root  / model_name / qtype
        answer_folder = answer_root  / model_name / qtype

        prompt_folder.mkdir(parents=True, exist_ok=True)
        premise_folder.mkdir(parents=True, exist_ok=True)
        answer_folder.mkdir(parents=True, exist_ok=True)

        question_files = sorted(
            question_folder.glob("*.txt"),
            key=lambda path: int(path.stem),
        )
        records = []
        # 把100题读出来并构造相关信息，以字典形式放在records内
        for question_file in question_files:
            prompt_path = prompt_folder / question_file.name
            premise_path = premise_folder / question_file.name
            answer_path = answer_folder / question_file.name
            outputs_exist = all(
                path.is_file()
                for path in (prompt_path, premise_path, answer_path)
            )
            if outputs_exist and not overwrite:
                continue

            logical_query = ast.literal_eval(
                question_file.read_text(encoding="utf-8")
            )
            e1, r1, e2, r2, e3, r3 = parse_logical_query(
                logical_query,
                qtype,
            )
            templates = prompt_generator.generate_prompt(
                logical_query,
                qtype,
            )
            records.append({
                "question_file": question_file,
                "prompt_path": prompt_path,
                "premise_path": premise_path,
                "answer_path": answer_path,
                "templates": templates,
                "entities": [e1, e2, e3],
                "relations": [r1, r2, r3],
                "answers": [],
                "answer_sets": [],
                "raw_answers": [],
                "confidences": [],
                "phase_prompts": {},
                "phase_premises": {},
                "phase_answers": {},
                "terminal_failed": False,
                "input_chars": [],
                "input_tokens": [],
            })

        if not records:
            print(f"/{qtype}/dynamic: already complete.")
            continue

        max_phase = max(len(record["templates"]) for record in records)
        for phase_index in range(max_phase):

            phase = phase_index + 1
            phase_key = str(phase)
            skipped_records = [
                record
                for record in records
                if (
                    phase in record["templates"]
                    and record["terminal_failed"]
                )
            ]
            for record in skipped_records:

                record["phase_prompts"][phase_key] = {
                    "template": record["templates"][phase],
                    "question": None,
                    "prompt": None,
                    "status": "skipped_previous_failure",
                }
                record["phase_premises"][phase_key] = {
                    "head_source": "none",
                    "head_entities": [],
                    "target_relation": None,
                    "operation": "skipped",
                    "triplets": [],
                    "status": "skipped_previous_failure",
                }
                record["answers"].append("None")
                record["input_chars"].append(None)
                record["input_tokens"].append(None)
                record["answer_sets"].append(set())
                record["raw_answers"].append(
                    "SKIPPED_PREVIOUS_FAILURE"
                )
                record["confidences"].append("failed")
                record["phase_answers"][phase_key] = {
                    "raw_answer": "SKIPPED_PREVIOUS_FAILURE",
                    "answer": "None",
                    "confidence": "failed",
                    "status": "skipped_previous_failure",
                }

            active_records = [
                record
                for record in records
                if (
                    phase in record["templates"]
                    and not record["terminal_failed"]
                )
            ]
            phase_prompts = []

            for record in active_records:
                template = record["templates"][phase]
                question = template
                if phase_index > 0:
                    question = swap_question_placeholders(
                        template,
                        record["answers"],
                    )
                previous_entities = (
                    record["answer_sets"][-1]
                    if record["answer_sets"]
                    else set()
                )
                premise_data = premise_generator.generate_phase_premise(
                    record["entities"],
                    record["relations"],
                    previous_entities,
                    phase,
                    qtype,
                )
                prompt = build_phase_prompt(question, premise_data)
                record["phase_prompts"][phase_key] = {
                    "template": template,
                    "question": question,
                    "prompt": prompt,
                    "status": "ready",
                }
                premise_data["status"] = "ready"
                record["phase_premises"][phase_key] = premise_data
                phase_prompts.append(prompt)

            raw_answers, input_stats = generate_batch(
                model,
                tokenizer,
                SYSTEM_PROMPT,
                phase_prompts,
                max_batch_size=max_batch_size,
                max_batch_tokens=max_batch_tokens,
                max_new_tokens=max_new_tokens,
                max_context_length=max_context_length,
                desc=f"{qtype}/dynamic/phase-{phase}",
            )

            for record, raw_answer, stats in zip(
                    active_records,
                    raw_answers,
                    input_stats,
            ):
                confidence, answer = parse_model_answer(raw_answer)
                if confidence == "failed":

                    answer = "None"
                    answer_entities = set()
                else:
                    answer_entities = llm_answer_to_set(answer)

                if raw_answer == "CONTEXT_OVERFLOW":
                    phase_status = "context_overflow"
                    record["terminal_failed"] = True
                elif raw_answer == "CUDA_OOM":
                    phase_status = "cuda_oom"
                    record["terminal_failed"] = True
                elif confidence == "failed":
                    phase_status = "parse_failed"
                else:
                    phase_status = "ok"

                record["answers"].append(answer)
                record["answer_sets"].append(answer_entities)
                record["raw_answers"].append(raw_answer)
                record["confidences"].append(confidence)
                record["input_chars"].append(stats["input_chars"])
                record["input_tokens"].append(stats["input_tokens"])
                record["phase_prompts"][phase_key]["status"] = phase_status
                record["phase_premises"][phase_key]["status"] = phase_status
                record["phase_answers"][phase_key] = {
                    "raw_answer": raw_answer,
                    "answer": answer,
                    "confidence": confidence,
                    "status": phase_status,
                }

        for record in records:
            save_dynamic_record(
                record,
                record["prompt_path"],
                record["premise_path"],
                record["answer_path"],
            )
def parse_args():
    parser = argparse.ArgumentParser(
        description="Online dynamic retrieval and staged KG inference."
    )
    parser.add_argument("--base-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
    gen_dynamic(
        adapter=args.adapter,
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

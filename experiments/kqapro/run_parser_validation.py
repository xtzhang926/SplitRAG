from typing import Any
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


QUERY_STRUCTS = ['1p', '2p', '2i']

SYSTEM_PROMPT = (
    """You are a semantic parser for knowledge graph questions.

Convert the user's question into a JSON structured query.

Query types:
- 1p: one anchor and one relation
- 2p: one anchor and two ordered relations
- 2i: two independent one-relation branches with intersection

Use this format:
{
  "type": "1p|2p|2i",
  "query": [
    {
      "anchor": "entity name",
      "path": [
        {
          "relation": "relation name",
          "direction": "forward|backward"
        }
      ]
    }
  ]
}

Keep entity and relation names in natural language.
For 2p, preserve relation order.
For 2i, output two independent branches.
Output JSON only, with no explanation or Markdown."""
)
def read_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def local_model(model_path, adapter=None, attn_implementation="sdpa"):
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
def build_dynamic_batches(lengths, max_batch_size, max_batch_tokens, max_new_tokens,):
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


def gen_local(
    adapter,
    base_dir,
    model_path,
    max_batch_size=16,
    max_batch_tokens=16000,
    max_new_tokens=512,
    max_context_length=40960,
    attn_implementation="sdpa",
):

    base_dir = Path(base_dir)
    answer_folder = base_dir / "answer"
    answer_folder.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    if adapter is None:
        raise ValueError("--adapter is required for fine-tuned inference.")
    adapter = Path(adapter)


    model, tokenizer = local_model(
        model_path,
        adapter,
        attn_implementation,
    )

    for qtype in QUERY_STRUCTS:
        prompt_folder = base_dir / "prompt"
        assert prompt_folder.is_dir(), f"{prompt_folder} does not exist."
        prompt_files = read_json(prompt_folder / f'{qtype}.json')
        records = []
        prompts = []
        for data in prompt_files:
            prompt = data['question']
            prompts.append(prompt)
            records.append({
                "queries": prompt,
                "answers": None,
                'sample_id':data['e11_sample_id'],
                'source_index':data['source_index']
            })
        raw_answers = generate_batch(
            model,
            tokenizer,
            SYSTEM_PROMPT,
            prompts,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            desc=f"{qtype}",
        )

        for record, raw_answer in zip(records,raw_answers,):
            record["answers"] = raw_answer
        write_json(
            answer_folder / f'{qtype}.json',
            records
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched staged inference for KG reasoning prompts."
    )
    parser.add_argument("--base_dir")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=16000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context-length", type=int, default=40960)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    return parser.parse_args()

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    gen_local(
        adapter=args.adapter,
        base_dir=args.base_dir,
        model_path=args.model_path,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()

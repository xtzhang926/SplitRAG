import hashlib
import gc
import itertools
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from typing import Any, Dict, List, Tuple, Set
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
"""
fine tuning
"""
ALLOWED_QUERY_TYPES = {"1p", "2p", "2i"}
ALLOWED_DIRECTIONS = {"forward", "backward"}
def normalize_name(text: str) -> str:

    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()
def parse_structured_output(text: str) -> Optional[Dict[str, Any]]:

    try:
        value = json.loads(str(text).strip())
    except (json.JSONDecodeError, TypeError):
        return None


    if not isinstance(value, dict):
        return None


    if set(value.keys()) != {"type", "query"}:
        return None

    qtype = value["type"]
    query = value["query"]


    if qtype not in ALLOWED_QUERY_TYPES:
        return None


    if not isinstance(query, list):
        return None


    if qtype == "1p":

        expected_branch_num = 1
        expected_path_lengths = [1]

    elif qtype == "2p":

        expected_branch_num = 1
        expected_path_lengths = [2]

    else:

        expected_branch_num = 2
        expected_path_lengths = [1, 1]

    if len(query) != expected_branch_num:
        return None


    for branch_index, branch in enumerate(query):


        if not isinstance(branch, dict):
            return None


        if set(branch.keys()) != {"anchor", "path"}:
            return None

        anchor = branch["anchor"]
        path = branch["path"]


        if not isinstance(anchor, str) or not anchor.strip():
            return None


        if not isinstance(path, list):
            return None


        if len(path) != expected_path_lengths[branch_index]:
            return None

        for step in path:

            if not isinstance(step, dict):
                return None


            if set(step.keys()) != {"relation", "direction"}:
                return None

            relation = step["relation"]
            direction = step["direction"]


            if not isinstance(relation, str) or not relation.strip():
                return None

            if direction not in ALLOWED_DIRECTIONS:
                return None

    return value
def normalized_branch(branch: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:

    anchor = normalize_name(branch["anchor"])

    path = tuple(
        (
            normalize_name(step["relation"]),
            step["direction"],
        )
        for step in branch["path"]
    )

    return anchor, path
def branch_pair_score(gold_branch: Dict[str, Any], pred_branch: Dict[str, Any],) -> int:

    gold_anchor, gold_path = normalized_branch(gold_branch)
    pred_anchor, pred_path = normalized_branch(pred_branch)

    score = int(gold_anchor == pred_anchor)

    # 当前合法 2i path 长度都为 1，但这里写成通用 zip。
    for (gold_relation, gold_direction), (pred_relation, pred_direction) in zip(
        gold_path,
        pred_path,
    ):
        score += int(gold_relation == pred_relation)
        score += int(gold_direction == pred_direction)

    return score
def align_predicted_branches(gold: Dict[str, Any], pred: Dict[str, Any], ) -> List[Optional[Dict[str, Any]]]:


    gold_query = gold["query"]
    pred_query = pred["query"]


    if len(gold_query) != len(pred_query):
        return [None] * len(gold_query)


    if gold["type"] in {"1p", "2p"}:
        return list(pred_query)

    best_perm = None
    best_score = -1

    for perm in itertools.permutations(pred_query):
        score = sum(
            branch_pair_score(gold_branch, pred_branch)
            for gold_branch, pred_branch in zip(gold_query, perm)
        )

        if score > best_score:
            best_score = score
            best_perm = perm

    return list(best_perm)
def score_one_parse(gold: Dict[str, Any], pred: Optional[Dict[str, Any]], ) -> Dict[str, float]:

    assert gold is not None, "Gold parser output does not satisfy the frozen schema."


    gold_anchor_total = len(gold["query"])
    gold_relation_total = sum(len(branch["path"]) for branch in gold["query"])


    if pred is None:
        return {
            "format_correct": 0.0,
            "type_correct": 0.0,
            "anchor_correct": 0.0,
            "anchor_total": float(gold_anchor_total),
            "relation_correct": 0.0,
            "relation_total": float(gold_relation_total),
            "direction_correct": 0.0,
            "direction_total": float(gold_relation_total),
            "relation_slot_correct": 0.0,
            "relation_slot_total": float(gold_relation_total),
            "exact_correct": 0.0,
        }


    format_correct = 1.0


    type_correct = float(pred["type"] == gold["type"])

    aligned_pred = align_predicted_branches(gold, pred)

    anchor_correct = 0
    relation_correct = 0
    direction_correct = 0
    relation_slot_correct = 0

    for gold_branch, pred_branch in zip(gold["query"], aligned_pred):


        if pred_branch is None:
            continue

        gold_anchor = normalize_name(gold_branch["anchor"])
        pred_anchor = normalize_name(pred_branch["anchor"])

        anchor_correct += int(gold_anchor == pred_anchor)

        gold_path = gold_branch["path"]
        pred_path = pred_branch["path"]


        for gold_step, pred_step in zip(gold_path, pred_path):

            gold_relation = normalize_name(gold_step["relation"])
            pred_relation = normalize_name(pred_step["relation"])

            relation_is_correct = gold_relation == pred_relation
            direction_is_correct = (
                gold_step["direction"] == pred_step["direction"]
            )

            relation_correct += int(relation_is_correct)
            direction_correct += int(direction_is_correct)


            relation_slot_correct += int(
                relation_is_correct and direction_is_correct
            )

    exact_correct = float(
        type_correct == 1.0
        and anchor_correct == gold_anchor_total
        and relation_correct == gold_relation_total
        and direction_correct == gold_relation_total
    )

    return {
        "format_correct": format_correct,
        "type_correct": type_correct,
        "anchor_correct": float(anchor_correct),
        "anchor_total": float(gold_anchor_total),
        "relation_correct": float(relation_correct),
        "relation_total": float(gold_relation_total),
        "direction_correct": float(direction_correct),
        "direction_total": float(gold_relation_total),
        "relation_slot_correct": float(relation_slot_correct),
        "relation_slot_total": float(gold_relation_total),
        "exact_correct": exact_correct,
    }
class GenerationParseMetricsCallback(TrainerCallback):


    def __init__(
        self,
        tokenizer,
        records,
        batch_size=2,
        max_new_tokens=256,
    ):
        self.tokenizer = tokenizer
        self.records = records
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        model=None,
        **kwargs,
    ):

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        was_training = model.training
        model.eval()


        sample_num = len(self.records)

        format_correct_sum = 0.0
        type_correct_sum = 0.0

        anchor_correct_sum = 0.0
        anchor_total_sum = 0.0

        relation_correct_sum = 0.0
        relation_total_sum = 0.0

        direction_correct_sum = 0.0
        direction_total_sum = 0.0

        relation_slot_correct_sum = 0.0
        relation_slot_total_sum = 0.0

        exact_correct_sum = 0.0


        qtype_exact_correct = {qtype: 0.0 for qtype in ALLOWED_QUERY_TYPES}
        qtype_sample_num = {qtype: 0 for qtype in ALLOWED_QUERY_TYPES}


        with torch.inference_mode():


            for start in range(0, sample_num, self.batch_size):

                batch = self.records[start:start + self.batch_size]

                prompt_texts = [
                    self.tokenizer.apply_chat_template(
                        record["messages"][:-1],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for record in batch
                ]

                encoded = self.tokenizer(
                    prompt_texts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )


                device = next(model.parameters()).device
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }


                outputs = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )



                input_length = encoded["input_ids"].shape[1]
                new_tokens = outputs[:, input_length:]

                texts = self.tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                )


                for record, generated_text in zip(batch, texts):


                    pred = parse_structured_output(generated_text)


                    gold_text = record["messages"][-1]["content"]
                    gold = parse_structured_output(gold_text)


                    if gold is None:
                        raise ValueError(
                            "Invalid gold parser JSON in validation data:\n"
                            f"{gold_text}"
                        )

                    one = score_one_parse(gold, pred)

                    format_correct_sum += one["format_correct"]
                    type_correct_sum += one["type_correct"]

                    anchor_correct_sum += one["anchor_correct"]
                    anchor_total_sum += one["anchor_total"]

                    relation_correct_sum += one["relation_correct"]
                    relation_total_sum += one["relation_total"]

                    direction_correct_sum += one["direction_correct"]
                    direction_total_sum += one["direction_total"]

                    relation_slot_correct_sum += one["relation_slot_correct"]
                    relation_slot_total_sum += one["relation_slot_total"]

                    exact_correct_sum += one["exact_correct"]

                    gold_qtype = gold["type"]
                    qtype_sample_num[gold_qtype] += 1
                    qtype_exact_correct[gold_qtype] += one["exact_correct"]


                del outputs
                del encoded


        generated_metrics = {

            "eval_valid_parse_rate":
                format_correct_sum / sample_num,


            "eval_type_accuracy":
                type_correct_sum / sample_num,


            "eval_anchor_accuracy":
                anchor_correct_sum / max(1.0, anchor_total_sum),


            "eval_relation_accuracy":
                relation_correct_sum / max(1.0, relation_total_sum),


            "eval_direction_accuracy":
                direction_correct_sum / max(1.0, direction_total_sum),


            "eval_relation_slot_accuracy":
                relation_slot_correct_sum / max(1.0, relation_slot_total_sum),


            "eval_exact_parse_accuracy":
                exact_correct_sum / sample_num,
        }


        for qtype in sorted(ALLOWED_QUERY_TYPES):
            count = qtype_sample_num[qtype]
            generated_metrics[f"eval_exact_parse_{qtype}"] = (
                qtype_exact_correct[qtype] / count
                if count > 0
                else 0.0
            )

        if metrics is not None:
            metrics.update(generated_metrics)


        if state.log_history:
            state.log_history[-1].update(generated_metrics)


        self.tokenizer.padding_side = old_padding_side

        if was_training:
            model.train()

        return control
class AnswerDataCollator:


    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):


        max_length = max(len(item["input_ids"]) for item in features)


        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_length = (
                (max_length + multiple - 1) // multiple
            ) * multiple

        input_ids = []
        attention_mask = []
        labels = []

        for item in features:

            pad_length = max_length - len(item["input_ids"])


            input_ids.append(
                item["input_ids"]
                + [self.tokenizer.pad_token_id] * pad_length
            )


            attention_mask.append(
                item["attention_mask"] + [0] * pad_length
            )


            labels.append(
                item["labels"] + [-100] * pad_length
            )

        return {
            "input_ids":
                torch.tensor(input_ids, dtype=torch.long),

            "attention_mask":
                torch.tensor(attention_mask, dtype=torch.long),

            "labels":
                torch.tensor(labels, dtype=torch.long),
        }
def set_fine_tuning_seed(seed: int) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
def apply_chat_template(tokenizer, messages, add_generation_prompt: bool, ):

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        return_dict=True,
    )
    return encoded["input_ids"]
def tokenize_record(record, tokenizer, max_length):

    messages = record["messages"]

    assert len(messages) == 3

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


    gold = parse_structured_output(messages[-1]["content"])
    assert gold is not None, (
        "Assistant gold output does not satisfy parser schema:\n"
        f"{messages[-1]['content']}"
    )

    prompt_ids = apply_chat_template(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
    )


    input_ids = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )


    assert prompt_ids == input_ids[:len(prompt_ids)], (
        "The generation prompt is not an exact prefix "
        "of the full conversation."
    )

    prompt_length = len(prompt_ids)


    labels = [-100] * prompt_length + input_ids[prompt_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,


        "length": len(input_ids),
        "valid": (
            len(input_ids) <= max_length
            and any(label != -100 for label in labels)
        ),
    }
def load_model(model_path):

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    compute_dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )


    model.config.use_cache = False

    return model, use_bf16
def add_lora_adapter(model, lora_rank, lora_alpha, lora_dropout, ):

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()
    return model
def save_train_config(save_path, train_config):
    with open(save_path / "train_config.json","w",encoding="utf-8",) as f:
        json.dump(train_config,f,ensure_ascii=False,indent=2,)
class FineTuningTool:

    def __init__(self, fine_tuning_dir):
        self.fine_tuning_dir = Path(fine_tuning_dir)

    def load_fine_tuning_data(self,tokenizer,max_length,):
        train_path = self.fine_tuning_dir / "parser_train.json"
        valid_path = self.fine_tuning_dir / "parser_valid.json"

        assert train_path.is_file(), f"{train_path} does not exist."
        assert valid_path.is_file(), f"{valid_path} does not exist."


        raw_dataset = load_dataset(
            "json",
            data_files={
                "train": str(train_path),
                "validation": str(valid_path),
            },
        )


        tokenized_dataset = raw_dataset.map(
            lambda record, index: {
                **tokenize_record(
                    record,
                    tokenizer,
                    max_length,
                ),
                "record_index": index,
            },
            with_indices=True,
            remove_columns=raw_dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Tokenizing parser dataset",
        )

        filter_statistics = {}
        validation_indexes = []

        for split in tokenized_dataset.keys():

            original_dataset = tokenized_dataset[split]
            original_num = len(original_dataset)

            dropped_lengths = [
                length
                for length, valid in zip(
                    original_dataset["length"],
                    original_dataset["valid"],
                )
                if not valid
            ]

            filtered_dataset = original_dataset.filter(
                lambda record: record["valid"],
                load_from_cache_file=False,
                desc=f"Filtering parser/{split}",
            )

            valid_num = len(filtered_dataset)
            dropped_num = original_num - valid_num

            assert valid_num > 0, (
                f"No valid samples remain in parser/{split}."
            )

            if split == "validation":
                validation_indexes = list(filtered_dataset["record_index"])

            kept_lengths = list(filtered_dataset["length"])

            filter_statistics[split] = {
                "original_num": original_num,
                "retained_num": valid_num,
                "dropped_num": dropped_num,
                "dropped_ratio": round(
                    dropped_num / original_num,
                    6,
                ),
                "max_retained_length": max(kept_lengths),
                "min_dropped_length": (
                    min(dropped_lengths)
                    if dropped_lengths
                    else None
                ),
                "max_dropped_length": (
                    max(dropped_lengths)
                    if dropped_lengths
                    else None
                ),
            }


            tokenized_dataset[split] = (
                filtered_dataset.remove_columns(
                    [
                        "length",
                        "valid",
                        "record_index",
                    ]
                )
            )

            print(
                f"parser/{split}: "
                f"{valid_num} retained, "
                f"{dropped_num} dropped."
            )

        validation_records = [
            raw_dataset["validation"][index]
            for index in validation_indexes
        ]

        assert (
            len(validation_records)
            == len(tokenized_dataset["validation"])
        )

        return (
            tokenized_dataset,
            validation_records,
            filter_statistics,
        )

    def finetune_model(
        self,
        model_path,
        model_name="Qwen3-4B",
        max_length=1024,
        epoch_num=3,
        learning_rate=2e-4,
        train_batch_size=1,
        valid_batch_size=2,
        gradient_accumulation_steps=16,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        seed=926,
        gradient_checkpointing=True,
        resume_from_checkpoint=None,
        generation_batch_size=2,
        generation_max_new_tokens=256,
    ):

        assert torch.cuda.is_available(), "CUDA GPU is required."


        set_fine_tuning_seed(seed)

        model_path = Path(model_path)
        assert model_path.is_dir(), f"{model_path} does not exist."


        save_path = self.fine_tuning_dir / "adapter" / model_name / "parser"
        save_path.mkdir(parents=True,exist_ok=True,)

        # ---------- tokenizer ----------
        tokenizer = AutoTokenizer.from_pretrained(model_path,)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token


        tokenizer.padding_side = "right"


        tokenized_dataset, validation_records, filter_statistics = self.load_fine_tuning_data(tokenizer,max_length,)


        model, use_bf16 = load_model(model_path)
        model = add_lora_adapter(model,lora_rank,lora_alpha,lora_dropout,)



        effective_batch_size = train_batch_size * gradient_accumulation_steps
        steps_per_epoch = math.ceil(len(tokenized_dataset["train"]) / effective_batch_size)
        total_steps = math.ceil(steps_per_epoch * epoch_num)

        warmup_steps = max(1,round(total_steps * 0.03),)


        train_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=epoch_num,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=valid_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False,},
            bf16=use_bf16,
            fp16=not use_bf16,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=10,


            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,

            load_best_model_at_end=True,
            metric_for_best_model="eval_exact_parse_accuracy",
            greater_is_better=True,
            report_to="none",
            logging_dir=save_path / "logs",
            remove_unused_columns=False,
            seed=seed,
            data_seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=AnswerDataCollator(tokenizer,),
            callbacks=[

                GenerationParseMetricsCallback(
                    tokenizer=tokenizer,
                    records=validation_records,
                    batch_size=generation_batch_size,
                    max_new_tokens=generation_max_new_tokens,
                ),

                EarlyStoppingCallback(early_stopping_patience=2,),
            ],
        )

        torch.cuda.reset_peak_memory_stats()
        train_start_time = time.perf_counter()


        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint,)

        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)


        train_config = {
            "task": "E11_predicted_parse",
            "model_name": model_name,
            "model_path": str(model_path),

            "max_length": max_length,
            "epoch_num": epoch_num,
            "learning_rate": learning_rate,

            "train_batch_size": train_batch_size,
            "valid_batch_size": valid_batch_size,
            "gradient_accumulation_steps":
                gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,

            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,

            "seed": seed,


            "optimization_loss":
                "assistant_only_token_cross_entropy",


            "best_model_metric":
                "eval_exact_parse_accuracy",

            "parser_metrics": [
                "valid_parse_rate",
                "type_accuracy",
                "anchor_accuracy",
                "relation_accuracy",
                "direction_accuracy",
                "relation_slot_accuracy",
                "exact_parse_accuracy",
                "exact_parse_1p",
                "exact_parse_2p",
                "exact_parse_2i",
            ],
        }

        train_seconds = (
            time.perf_counter()
            - train_start_time
        )

        train_config["training_resource"] = {
            "gpu_name":
                torch.cuda.get_device_name(0),

            "train_seconds":
                round(train_seconds, 2),

            "peak_cuda_allocated_gb":
                round(
                    torch.cuda.max_memory_allocated()
                    / 1024 ** 3,
                    3,
                ),

            "global_step":
                trainer.state.global_step,

            "best_checkpoint":
                trainer.state.best_model_checkpoint,

            "best_eval_exact_parse_accuracy":
                trainer.state.best_metric,

            "data_filter":
                filter_statistics,
        }

        save_train_config(
            save_path,
            train_config,
        )


        del trainer
        del model

        gc.collect()
        torch.cuda.empty_cache()

        print("Parser fine-tuning finished.")
        print(f"Saved to: {save_path}")

        return save_path
def run_fine_tune():


    fine_tuning_dir = Path(r"")

    base_model = Path(r"")
    tool = FineTuningTool(fine_tuning_dir=fine_tuning_dir,)
    tool.finetune_model(
        model_path=base_model,
        model_name="Qwen3-4B",

        max_length=512,

        epoch_num=3,
        learning_rate=2e-4,


        train_batch_size=4,
        valid_batch_size=8,
        gradient_accumulation_steps=4,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        seed=926,
        gradient_checkpointing=True,
        resume_from_checkpoint=None,

        generation_batch_size=4,

        generation_max_new_tokens=256,
    )



if __name__ == '__main__':
    pass

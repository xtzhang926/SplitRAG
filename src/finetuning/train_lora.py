import json
import os.path
import pathlib
import random
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
original_read_text = pathlib.Path.read_text
def utf8_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return original_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = utf8_read_text
from trl import SFTTrainer, SFTConfig
def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
def get_all_filenames(path):
    """获取路径下所有文件名"""
    all_items = os.listdir(path)  # 获取目录所有条目
    return all_items
def load_data(task,base_train):
    qtype_list = ['1p', '2p', '2i', '3i', '2u']
    data = []
    for qtype in qtype_list:
        path = os.path.join(base_train, task, 'prompt', qtype)
        file_num = len(get_all_filenames(path))
        for i in range(1, file_num + 1):
            with open(os.path.join(path, f"{i}.txt"), "r", encoding='utf-8') as f:
                data.append(json.loads(f.read()))
    random.shuffle(data)
    return data
def train_llama(data,local_model_path,output_dir):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    tokenizer.model_max_length = 512
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    dataset = Dataset.from_list(data)

    def format_prompt(example):
        if example.get("input", "").strip():
            user_content = f"{example['instruction']}\n{example['input']}"
        else:
            user_content = example['instruction']

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=2,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_steps=50,
        gradient_checkpointing=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,               # 这里传入的是纯粹的 base model
        train_dataset=dataset,
        peft_config=lora_config,   # 这里传入配置，让 Trainer 自动包LoRA
        formatting_func=format_prompt,
        args=training_args,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
def train_qwen(data,local_model_path,output_dir):

    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    dataset = Dataset.from_list(data)



    def format_prompt(example):
        if example.get("input", "").strip():
            user_content = f"{example['instruction']}\n{example['input']}"
        else:
            user_content = example['instruction']

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=2,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_steps=50,
        gradient_checkpointing=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_prompt,
        args=training_args,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
def model_eval(data,lora_model_path,lora_model_list,output_dir):


    max_seq_length = 512
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(data)
    def preprocess_function(examples):

        prompts = []
        for instruction, input_str in zip(examples["instruction"], examples["input"]):
            if input_str:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            prompts.append(prompt)
        full_texts = [prompt + output for prompt, output in zip(prompts, examples["output"])]
        model_inputs = tokenizer(full_texts, max_length=max_seq_length, truncation=True, padding="max_length")
        prompt_inputs = tokenizer(prompts, max_length=max_seq_length, truncation=True, padding=False)

        labels = []
        for i in range(len(full_texts)):
            prompt_len = len(prompt_inputs["input_ids"][i])
            input_ids = model_inputs["input_ids"][i]
            # 5. 构建Label：将Prompt部分和Padding部分的label设为-100，仅对Output部分计算Loss
            label = [
                token_id if (idx >= prompt_len and token_id != tokenizer.pad_token_id) else -100
                for idx, token_id in enumerate(input_ids)
            ]
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs
    eval_dataset = dataset.map(preprocess_function, batched=True)
    best_model_path = None
    best_eval_loss = float("inf")
    eval_results = {}
    for lora_path in lora_model_list:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=8,
            dataloader_drop_last=False,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            eval_dataset=eval_dataset,
        )
        metrics = trainer.evaluate()
        current_loss = metrics.get("eval_loss", float("inf"))
        eval_results[lora_path] = current_loss

        if current_loss < best_eval_loss:
            best_eval_loss = current_loss
            best_model_path = lora_path

        del base_model
        del model
        del trainer
        torch.cuda.empty_cache()

    for path, loss in eval_results.items():
        print(f"model: {path} | loss: {loss:.4f}")
    print(f"\nbest model: {best_model_path}，loss: {best_eval_loss:.4f}")
if __name__ == '__main__':
    set_global_seed(0)
    task_list = ["train", "valid", "test"]
    data = load_data(task_list[0],base_train)
    train_qwen(data,model_path,output)


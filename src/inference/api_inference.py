import ast
import json
import os
import random
import re
from pathlib import Path
import torch
from openai import OpenAI


def swap_question_placeholders(enhance_query, llm_answer):
    """将问题中的占位符替换为对应的问题答案"""
    if "[PP1]" in enhance_query:
        sai = llm_answer[0]
        enhance_query = enhance_query.replace("[PP1]", "{" + sai + "}")
    if "[PP2]" in enhance_query:
        sai = llm_answer[1]
        enhance_query = enhance_query.replace("[PP2]", "{" + sai + "}")
    if "[PP3]" in enhance_query:
        sai = llm_answer[2]
        enhance_query = enhance_query.replace("[PP3]", "{" + sai + "}")
    return enhance_query
def generate_api(task,model):
    def call_api(system_prompt, prompt, model_name):

        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.05,
            max_tokens=2048,
            extra_body = {"enable_thinking": False},
            timeout=120,
        )
        content = completion.choices[0].message.content
        return content
    def load_single_stage_prompts(task, qtype):
        folder = PROMPT_BASE / task / qtype
        results = []
        files = sorted(os.listdir(folder), key=lambda x: int(x.replace(".txt", "")))
        for fname in files:
            idx = int(fname.replace(".txt", ""))
            filepath = folder / fname
            with open(filepath, "r", encoding="utf-8") as f:
                data = ast.literal_eval(f.read())
            prompt_text = data["query"]
            true_answer = data["answer"]
            results.append((idx, prompt_text, true_answer))
        return results
    def load_splitrag_prompts(qtype):

        folder = PROMPT_BASE / "step_rag" / qtype
        results = []
        files = sorted(os.listdir(folder), key=lambda x: int(x.replace(".txt", "")))
        for fname in files:
            idx = int(fname.replace(".txt", ""))
            filepath = folder / fname

            with open(filepath, "r", encoding="utf-8") as f:
                data = ast.literal_eval(f.read())

            premise_list = data['query']["premise"]  # list of str, 每个子问题的上下文
            question_info = data['query']["question"]
            question_tag = question_info["question_tag"]
            question_list = question_info["question"][qtype]  # list of str
            explain_tag = question_info["explain_tag"]
            true_answer = data["answer"]
            # 组装每个子问题的完整 prompt
            sub_prompts = []
            for pz, qz in zip(premise_list, question_list):
                sub_prompts.append("".join([pz, question_tag, qz, explain_tag]))
            results.append((idx, sub_prompts, true_answer))
        return results
    def process_abstract(system_prompt,query_sample,model_name,qtype):
        for (idx,query,answer) in tqdm(query_sample, total=len(query_sample), desc=f"{qtype} Querying"):
            llm_ans = call_api(system_prompt,query,model_name)
            json_match = re.search(r'\{.*}', llm_ans, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    ans = set(data["answer"])
                except Exception:

                    ans = set()
            else:
                ans = set()
            llm_ans = ",".join(list(ans))
            llm_path = os.path.join(LLM_ANS_BASE, model_name, "abstract", f"{qtype}")
            os.makedirs(llm_path, exist_ok=True)
            ans_file = os.path.join(llm_path, f"{idx}.txt")
            with open(ans_file, "w", encoding="utf-8") as f:
                f.write("@\n@".join([llm_ans, answer]))
    def process_FewShot_CoT(system_prompt,query_sample,model_name,qtype):
        for (idx,query,answer) in tqdm(query_sample, total=len(query_sample), desc=f"{qtype} Querying"):
            raw_llm_ans = call_api(system_prompt,query,model_name)
            pattern = r'"answer":\s*(\[.*?\])'
            try:
                matches = re.findall(pattern, raw_llm_ans)
                if matches:
                    match = matches[-1]
                    llm_answer = ",".join(set(json.loads(match)))
                else:
                    llm_answer = ""
            except:
                llm_answer = ""

            llm_path = os.path.join(LLM_ANS_BASE, model_name, "FewShot_CoT", f"{qtype}")
            llm_path_p1 = os.path.join(LLM_ANS_BASE, model_name, "FewShot_CoT_p1", f"{qtype}")
            os.makedirs(llm_path, exist_ok=True)
            os.makedirs(llm_path_p1, exist_ok=True)
            ans_file = os.path.join(llm_path, f"{idx}.txt")
            ans_file_p1 = os.path.join(llm_path_p1, f"{idx}.txt")
            with open(ans_file, "w", encoding="utf-8") as f:
                f.write("@\n@".join([raw_llm_ans, answer]))
            with open(ans_file_p1, "w", encoding="utf-8") as f:
                f.write("@\n@".join([llm_answer, answer]))
    def process_splitrag(system_prompt,query_sample,model_name,qtype):
        for (idx,query_list,answer) in tqdm(query_sample, total=len(query_sample), desc=f"{qtype} Querying"):
            llm_answer = []
            for id, enhance_query in enumerate(query_list):
                if id == 0:
                    r = call_api(system_prompt,enhance_query,model_name)

                else:
                    change_query = swap_question_placeholders(enhance_query, llm_answer)
                    r = call_api(system_prompt, change_query, model_name)
                json_match = re.search(r'\{.*}', r, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        data = json.loads(json_str)
                        ans = set(data["answer"])
                    except Exception:

                        ans = set()
                else:
                    ans = set()
                llm_ans = ",".join(list(ans))
                llm_answer.append(llm_ans)
            llm = llm_answer[-1]
            llm_path = os.path.join(LLM_ANS_BASE, model_name, "step_rag", f"{qtype}")
            os.makedirs(llm_path, exist_ok=True)
            ans_file = os.path.join(llm_path, f"{idx + 1}.txt")
            with open(ans_file, "w", encoding="utf-8") as f:
                f.write("@\n@".join([llm, answer]))
    API_KEY = ""
    BASE_URL = ""
    PROMPT_BASE = Path(r"")
    LLM_ANS_BASE = Path(r"")
    QTYPES = ["2i", "2u", "3i"]

    if task in ["abstract", "step_rag"]:
        system_prompt = """You are an expert in deductive reasoning. 
                                        Your response must be ONLY a valid JSON object with the following exact structure:
                                        {"answer": ["string", "string"]}
                                        Do not include any other text, code fences, or explanations."""
    elif task == "FewShot_CoT":
        system_prompt = """You are an expert in deductive reasoning.
            First, reason step by step. Then output a valid JSON object with the following exact structure:
            {"answer": ["string", "string"]}"""
    else:
        raise "error"
    SAMPLE_SIZE = 50
    random.seed(42)
    for qtype in QTYPES:
        # import query
        if task in ["abstract", "FewShot_CoT"]:
            query = load_single_stage_prompts(task, qtype)
        elif task == "step_rag":
            query = load_splitrag_prompts(qtype)
        else:
            raise "error"
        query_sample = random.sample(query, SAMPLE_SIZE)
        if task == 'abstract':
            process_abstract(system_prompt,query_sample,model,qtype)
        elif task == 'FewShot_CoT':
            process_FewShot_CoT(system_prompt, query_sample, model, qtype)
        elif task == 'step_rag':
            process_splitrag(system_prompt, query_sample, model, qtype)
        else:
            raise "error"
def main():
    models = ["deepseek-v4-pro", "qwen3.7-max", "glm-4.7"]
    task_list = ["abstract", "step_rag", "FewShot_CoT"]
    for task in task_list:
        for model in models:
            generate_api(task,model)




if __name__ == "__main__":

    main()




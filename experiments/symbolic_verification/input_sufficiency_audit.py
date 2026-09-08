import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANSWER_MARKER = "Answer the question:"
TRIPLET_PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)
QUESTION_ENTITY_PATTERN = re.compile(
    r"(connected\s+to(?:\s+entity)?\s+)(\d+)",
    flags=re.IGNORECASE,
)
QUESTION_RELATION_PATTERN = re.compile(
    r"(by\s+relation\s+)(\d+)",
    flags=re.IGNORECASE,
)
OPERATION = {
    '1p':['p'],
    '2p':['p','2p'],
    '2i':['p','p','2i'],
    '3i':['p','p','p','3i'],
    '2u':['p','p','u'],
}
QUESTION_MARKER = "\n\nAnswer the question:\n"
def p(triplets, entity_ids, relation_ids):
    ans = set()
    for triplet in triplets:
        if triplet[0] in entity_ids and triplet[1] in relation_ids:
            ans.add(triplet[2])
    return ans
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def get_msg(text):
    triplet_section, _, question_section = split_prompt(text)

    triplets = [
        (int(head), int(relation), int(tail))
        for head, relation, tail in TRIPLET_PATTERN.findall(triplet_section)
    ]

    entity_ids = [
        int(match.group(2))
        for match in QUESTION_ENTITY_PATTERN.finditer(question_section)
    ]

    relation_ids = [
        int(match.group(2))
        for match in QUESTION_RELATION_PATTERN.finditer(question_section)
    ]
    return triplets, entity_ids, relation_ids
def split_prompt(text):
    if ANSWER_MARKER not in text:
        raise ValueError(f"Prompt miss：{ANSWER_MARKER!r}")
    triplet_section, question_section = text.split(ANSWER_MARKER, maxsplit=1)
    return triplet_section, ANSWER_MARKER, question_section
QUERY_STRUCTS = ['1p', '2p', '2i', '3i', '2u']
def proof_evidence_upper(prompt_dir,task,save_dir):
    all_num = 0
    true_num = 0
    for qtype in QUERY_STRUCTS:
        operation_list = OPERATION[qtype]
        files = list((prompt_dir / task / qtype).glob("*.txt"))
        for file in files:

            ans_list = []
            text = read_json(file)
            for phase, query in enumerate(text['query']):
                operation = operation_list[phase]
                triplets, entity_id, relation_id = get_msg(query)
                if operation == 'p':
                    ans_list.append(p(triplets, entity_id, relation_id))
                elif operation == '2p':
                    ans_list.append(p(triplets, ans_list[0], relation_id))
                elif operation == '2i':
                    ans_list.append(ans_list[0].intersection(ans_list[1]))
                elif operation == '3i':
                    ans_list.append(ans_list[0].intersection(ans_list[1]).intersection(ans_list[2]))
                else:
                    assert operation == 'u'
                    ans_list.append(ans_list[0].union(ans_list[1]))
            gold = set(map(int, text['answer'].strip().split(",")))
            result = {
                'executor': str(ans_list),
                'gold': str(gold),
                'result': True if gold == ans_list[-1] else False
            }
            all_num += 1
            if gold == ans_list[-1]:
                true_num+=1
            write_json(
                save_dir / task / qtype / file.name,
                result
            )
def eval_result(base_dir,model,score_dir):

    for qtype in QUERY_STRUCTS:
        files = list((base_dir / model / qtype).glob("*.txt"))
        for file in files:
            text = read_json(file)
            pred = set(text['pred'])
            gold = set(text['gold'])
            true_positive = len(pred & gold)
            precision = true_positive / len(pred) if pred else 0.0
            recall = true_positive / len(gold) if gold else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            result = {
                "exact_match": int(pred == gold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "missing_entities": sorted(gold - pred),
                "extra_entities": sorted(pred - gold),
            }
            write_json(
                score_dir / model / qtype / file.name,
                result
            )





if __name__ == '__main__':
    prompt_dir = Path('')
    save_dir = Path('')
    score_dir = Path('')
    base_dir = Path('')
    task = 'Decompose'
    # task = 'SplitRag'
    proof_evidence_upper(prompt_dir,task,save_dir)
    eval_result(base_dir,'Qwen3.5-0.8B',score_dir)


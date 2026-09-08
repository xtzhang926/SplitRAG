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
    """以 UTF-8 读取 JSON。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:
    """以 UTF-8 保存 JSON，并保持中文可读。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def get_msg(text):
    """提取三元组，以及问题文本中的实体 ID 和关系 ID。"""
    triplet_section, _, question_section = split_prompt(text)

    triplets = [
        (int(head), int(relation), int(tail))
        for head, relation, tail in TRIPLET_PATTERN.findall(triplet_section)
    ]

    # 实体必须来自问题描述，不从三元组推导。
    entity_ids = [
        int(match.group(2))
        for match in QUESTION_ENTITY_PATTERN.finditer(question_section)
    ]

    # 关系同样从问题描述提取；该规则包括 [PPn] 后的关系。
    relation_ids = [
        int(match.group(2))
        for match in QUESTION_RELATION_PATTERN.finditer(question_section)
    ]
    return triplets, entity_ids, relation_ids
def split_prompt(text):
    """将 prompt 分成三元组区域、标记文本和问题区域。"""
    if ANSWER_MARKER not in text:
        raise ValueError(f"Prompt 中缺少标记：{ANSWER_MARKER!r}")
    triplet_section, question_section = text.split(ANSWER_MARKER, maxsplit=1)
    return triplet_section, ANSWER_MARKER, question_section
QUERY_STRUCTS = ['1p', '2p', '2i', '3i', '2u']
def retrieved_evidence_upper_bound(dynamic_premise_dir,model,gold_dir,save_dir):
    for qtype in QUERY_STRUCTS:
        operation_list = OPERATION[qtype]
        files = list((dynamic_premise_dir / model / qtype).glob("*.txt"))
        for file in files:
            ans_list = []
            text = read_json(file)
            for phase, msg in text.items():
                operation = operation_list[int(phase)-1]
                triplets = [tuple(triple) for triple in msg['triplets']]
                entity_id = msg['head_entities']
                relation_id = [msg['target_relation']]
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
            gold_path = gold_dir / qtype / 'answer_abstract' / file.name
            gold = set(map(int, gold_path.read_text().strip().strip('{').strip('}').split(",")))
            result = {
                'executor': str(ans_list),
                'pred':list(ans_list[-1]),
                'gold': list(gold),
            }
            write_json(
                save_dir / model / qtype / file.name,
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
    dynamic_premise_dir = Path(r"")
    gold_dir = Path(r"")
    save_dir = Path(r"")
    base_dir = Path(r"")
    score_dir = Path(r"")
    task = 'Decompose'
    retrieved_evidence_upper_bound(dynamic_premise_dir,'Qwen3.5-0.8B',gold_dir,save_dir)
    eval_result(base_dir,'Qwen3.5-0.8B',score_dir)


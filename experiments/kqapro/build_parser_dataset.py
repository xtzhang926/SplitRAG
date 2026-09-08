import itertools
import re
import hashlib
import unicodedata

from typing import Any, Dict, List, Tuple, Set, Optional
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
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
"""data generate"""
TARGET_QTYPES = ("1p", "2p", "2i")
def load_source_data(label,qtype,base_dir):
    if label not in ['train','val']:
        raise ValueError("label wrong")
    file_path = base_dir / f'{qtype}.json'
    data = read_json(file_path)
    return data
def read_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def format_output(qtype,anchors,relations):
    if qtype == '1p':
        e = anchors[0]['name']
        r = relations[0][0]['predicate']
        d = relations[0][0]['direction']
        return {
            'type':'1p',
            'query':[
                {
                    "anchor": e,
                    "path": [
                        {
                            'relation': r,
                            'direction': d
                        }
                    ],
                },
            ]

        }
    elif qtype == '2p':
        e = anchors[0]['name']
        r1 = relations[0][0]['predicate']
        d1 = relations[0][0]['direction']
        r2 = relations[0][1]['predicate']
        d2 = relations[0][1]['direction']
        return {
            'type':'2p',
            'query':[
                {
                    "anchor": e,
                    "path": [
                        {
                            'relation': r1,
                            'direction': d1
                        },
                        {
                            'relation': r2,
                            'direction': d2
                        }
                    ],
                },
            ]

        }
    elif qtype == '2i':
        e1 = anchors[0]['name']
        r1 = relations[0][0]['predicate']
        d1 = relations[0][0]['direction']

        e2 = anchors[1]['name']
        r2 = relations[1][0]['predicate']
        d2 = relations[1][0]['direction']

        return {
            'type':'2i',
            'query':[
                {
                    "anchor": e1,
                    "path": [
                        {
                            'relation': r1,
                            'direction': d1
                        }
                    ],
                },
                {
                    "anchor": e2,
                    "path": [
                        {
                            'relation': r2,
                            'direction': d2
                        }
                    ],
                },
            ]

        }
    else:
        raise ValueError('error')
def set_rng(label,seed):
    value = f"{seed}|label={label}|split dataset|"
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
def construct_train(base_dir,save_dir):
    result = []
    rng = set_rng('split_data', 926)
    for qtype in TARGET_QTYPES:
        datas = load_source_data('train',qtype,base_dir)
        for data in datas:
            question = data['question']
            anchors = data['anchors']
            relations = data['relations']
            output = format_output(qtype,anchors,relations)
            result.append({
                'messages':[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
                ],
                "qtype":qtype,
                'sample_id':data['e11_sample_id'],
                'source_index':data['source_index'],
            })
    # result = result[:100]
    rng.shuffle(result)
    num = int(len(result)*0.9)
    write_json(
        save_dir / f'parser_train.json',
        result[:num]
    )
    write_json(
        save_dir / f'parser_valid.json',
        result[num:]
    )
def construct_gold_parse(base_dir,save_dir):
    for qtype in TARGET_QTYPES:
        result = []
        datas = load_source_data('val',qtype,base_dir)
        for data in datas:
            anchors = data['anchors']
            relations = data['relations']
            output = format_output(qtype,anchors,relations)
            result.append({
                'gold':json.dumps(output, ensure_ascii=False),
                'sample_id':data['e11_sample_id'],
                'source_index':data['source_index'],
            })
        write_json(
            save_dir / f'{qtype}.json',
            result
        )

"""eval"""
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
            direction_is_correct = gold_step["direction"] == pred_step["direction"]
            relation_correct += int(relation_is_correct)
            direction_correct += int(direction_is_correct)
            # Relation-slot Accuracy 要求 relation 和 direction 同时正确。
            relation_slot_correct += int(relation_is_correct and direction_is_correct)


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
def eval_parser():
    base_dir = Path(r"")
    for qtype in ALLOWED_QUERY_TYPES:
        golds = read_json(base_dir / 'gold' / f'{qtype}.json')
        preds = read_json(base_dir / 'parser_answer' / f'{qtype}.json')
        results = []
        for gold,pred in zip(golds,preds):
            assert gold['sample_id'] == pred['sample_id'] and gold['source_index'] == pred['source_index']
            strict_gold = parse_structured_output(gold['gold'])
            strict_pred = parse_structured_output(pred['answers'])
            score = score_one_parse(strict_gold,strict_pred)
            results.append(score)
        write_json(
            base_dir / 'score' / f'{qtype}.json',
            results
        )






if __name__ == '__main__':
    eval_parser()

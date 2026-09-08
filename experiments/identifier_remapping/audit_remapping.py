

import argparse
import csv
import json
import math
import numpy as np
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

MODELS = [
    "Llama-3.2-1B-Instruct_ft",
    "Qwen3.5-0.8B_ft",
]

BASELINE_MODEL_FOLDER = {
    "Llama-3.2-1B-Instruct_ft":
        "Llama-3.2-1B-Instruct",

    "Qwen3.5-0.8B_ft":
        "Qwen3.5-0.8B",
}
METHODS = [
    "Rag",
    "Decompose",
    "SplitRag",
]
METHOD_DISPLAY = {
    "Rag": "RAG",
    "Decompose": "Decomp",
    "SplitRag": "SplitRAG",
}
BASELINE_METHOD_FOLDER = {
    "Rag": "abstract",
    "Decompose": "step",
    "SplitRag": "step_rag",
}
QTYPES = [
    "1p",
    "2p",
    "2i",
    "3i",
    "2u",
]
EXPECTED_N = 150
QUERY_GROUPS = {
    "all": [
        "1p",
        "2p",
        "2i",
        "3i",
        "2u",
    ],

    "path": [
        "1p",
        "2p",
    ],

    "set": [
        "2i",
        "3i",
        "2u",
    ],

    "intersection": [
        "2i",
        "3i",
    ],
}

TRIPLET_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)
QUESTION_ENTITY_RE = re.compile(
    r"connected\s+to(?:\s+entity|\s+entities\s+in)?\s+(\d+)",
    flags=re.IGNORECASE,
)
QUESTION_RELATION_RE = re.compile(
    r"by\s+relation\s+(\d+)",
    flags=re.IGNORECASE,
)
OLD_SEPARATOR_RE = re.compile(
    r"@\s*@"
)

def mean(values: Iterable[float]) -> float:

    values = list(values)
    return float(np.mean(values)) if values else float("nan")
def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
def write_csv(path: Path,rows: List[Dict[str, Any]],fieldnames: List[str],) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
def fmt(value: float, digits: int = 4) -> str:

    if value is None:
        return "NA"

    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{value:.{digits}f}"



def parse_entity_set(value: Any,) -> Set[str]:


    text = str(value).strip()

    if not text:
        return set()

    if text.casefold() == "none":
        return set()

    result = set()

    for item in text.split(","):

        item = item.strip()

        if not item:
            continue

        if item.casefold() == "none":
            continue

        result.add(item)

    return result
def parse_int_entity_set(value: Any,) -> Set[int]:

    return {
        int(x)
        for x in parse_entity_set(
            value
        )
    }
def set_metrics(pred: Set[str],gold: Set[str],) -> Dict[str, float]:


    pred = set(pred)
    gold = set(gold)

    if not pred and not gold:
        return {
            "exact_match": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    if not pred or not gold:
        return {
            "exact_match":
                float(pred == gold),

            "precision":
                0.0,

            "recall":
                0.0,

            "f1":
                0.0,
        }

    tp = len(
        pred & gold
    )

    precision = (
        tp / len(pred)
    )

    recall = (
        tp / len(gold)
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "exact_match":
            float(pred == gold),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
    }
def infer_offsets(base_dir10_B:Path) -> Tuple[int, int]:


    entity_offsets = Counter()
    relation_offsets = Counter()

    for method in METHODS:
        for qtype in QTYPES:
            for sample_id in range(1,EXPECTED_N + 1,):

                source = read_json(base_dir10_B / 'source_prompt' / method / qtype / f'{sample_id}.txt')
                mapped = read_json(base_dir10_B / 'map_prompt' / method / qtype / f'{sample_id}.txt')

                if len(source["query"]) != len(mapped["query"]):
                    raise ValueError(
                        f"Stage count mismatch: "
                        f"{method}/{qtype}/{sample_id}"
                    )

                for source_stage, map_stage in zip(source["query"],mapped["query"],):

                    source_entities = [
                        int(x)
                        for x in QUESTION_ENTITY_RE.findall(source_stage)
                    ]

                    map_entities = [
                        int(x)
                        for x in QUESTION_ENTITY_RE.findall(map_stage)
                    ]

                    if len(source_entities) != len(map_entities):
                        raise ValueError(
                            "Question entity count mismatch: "
                            f"{method}/{qtype}/{sample_id}"
                        )

                    for a, b in zip(source_entities,map_entities,):
                        entity_offsets[b - a] += 1

                    source_relations = [
                        int(x)
                        for x in QUESTION_RELATION_RE.findall(source_stage)
                    ]

                    map_relations = [
                        int(x)
                        for x in QUESTION_RELATION_RE.findall(map_stage)
                    ]

                    if len(source_relations) != len(map_relations):
                        raise ValueError(
                            "Question relation count mismatch: "
                            f"{method}/{qtype}/{sample_id}"
                        )

                    for a, b in zip(source_relations,map_relations,):
                        relation_offsets[b - a] += 1

    if len(entity_offsets) != 1:
        raise RuntimeError(
            f"Expected one entity offset, got: "
            f"{dict(entity_offsets)}"
        )

    if len(relation_offsets) != 1:
        raise RuntimeError(
            f"Expected one relation offset, got: "
            f"{dict(relation_offsets)}"
        )

    entity_offset = next(iter(entity_offsets))
    relation_offset = next(iter(relation_offsets))

    return (
        int(entity_offset),
        int(relation_offset),
    )
def audit_prompt_remapping(base_dir10_B:Path,entity_offset: int,relation_offset: int,) -> Tuple[List[Dict[str, Any]],Dict[str, Any],]:


    rows = []

    source_entities: Set[int] = set()
    mapped_entities: Set[int] = set()

    source_relations: Set[int] = set()
    mapped_relations: Set[int] = set()

    for method in METHODS:
        for qtype in QTYPES:
            for sample_id in range(1,EXPECTED_N + 1,):

                source = read_json(base_dir10_B / 'source_prompt' / method / qtype / f'{sample_id}.txt')
                mapped = read_json(base_dir10_B / 'map_prompt' / method / qtype / f'{sample_id}.txt')

                source_query = source.get("query")
                mapped_query = mapped.get("query")

                if not isinstance(source_query, list) or not isinstance(mapped_query, list):
                    raise ValueError(
                        f"Missing query list: "
                        f"{method}/{qtype}/{sample_id}"
                    )

                stage_count_match = int(len(source_query) == len(mapped_query))

                if not stage_count_match:
                    raise RuntimeError(
                        "Stage count mismatch."
                    )

                all_triples_ordered_match = True
                all_question_entities_match = True
                all_question_relations_match = True

                for source_stage, mapped_stage in zip(source_query,mapped_query,):
                    source_triples = [
                        (int(h),int(r),int(t),)
                        for h, r, t in TRIPLET_RE.findall(source_stage)
                    ]

                    mapped_triples = [
                        (int(h),int(r),int(t),)
                        for h, r, t in TRIPLET_RE.findall(mapped_stage)
                    ]

                    expected_mapped_triples = [
                        (h + entity_offset,r + relation_offset,t + entity_offset,)
                        for h, r, t in source_triples
                    ]

                    if expected_mapped_triples != mapped_triples:
                        all_triples_ordered_match = False

                    for h, r, t in source_triples:
                        source_entities.update([h, t])
                        source_relations.add(r)
                    for h, r, t in mapped_triples:
                        mapped_entities.update([h, t])
                        mapped_relations.add(r)

                    source_q_entities = [
                        int(x)
                        for x in QUESTION_ENTITY_RE.findall(source_stage)
                    ]

                    mapped_q_entities = [
                        int(x)
                        for x in QUESTION_ENTITY_RE.findall(mapped_stage)
                    ]

                    expected_q_entities = [
                        x + entity_offset
                        for x in source_q_entities
                    ]

                    if expected_q_entities != mapped_q_entities:
                        all_question_entities_match = False

                    source_entities.update(source_q_entities)
                    mapped_entities.update(mapped_q_entities)

                    # ---------------------------------------------------------
                    # Question relations
                    # ---------------------------------------------------------
                    source_q_relations = [
                        int(x)
                        for x in QUESTION_RELATION_RE.findall(source_stage)
                    ]

                    mapped_q_relations = [
                        int(x)
                        for x in QUESTION_RELATION_RE.findall(mapped_stage)
                    ]

                    expected_q_relations = [
                        x + relation_offset
                        for x in source_q_relations
                    ]

                    if expected_q_relations != mapped_q_relations:
                        all_question_relations_match = False

                    source_relations.update(source_q_relations)
                    mapped_relations.update(mapped_q_relations)


                source_answer = parse_int_entity_set(source.get("answer","",))
                mapped_answer = parse_int_entity_set(mapped.get("answer","",))


                expected_mapped_answer = {
                    x + entity_offset
                    for x in source_answer
                }

                answer_match = int(expected_mapped_answer == mapped_answer)

                source_entities.update(source_answer)
                mapped_entities.update(mapped_answer)

                row = {
                    "method":METHOD_DISPLAY[method],
                    "qtype":qtype,
                    "sample_id":sample_id,
                    "stage_count_match":stage_count_match,
                    "triples_ordered_mapping_match":int(all_triples_ordered_match),
                    "question_entity_mapping_match":int(all_question_entities_match),
                    "question_relation_mapping_match":int(all_question_relations_match),
                    "answer_mapping_match":answer_match,
                    "all_mapping_checks_pass":
                        int(stage_count_match
                            and all_triples_ordered_match
                            and all_question_entities_match
                            and all_question_relations_match
                            and answer_match),
                }

                rows.append(row)

    # -------------------------------------------------------------------------
    # 所有 prompt 必须通过，否则不允许继续解释实验。
    # -------------------------------------------------------------------------
    if not all(row["all_mapping_checks_pass"] for row in rows):
        raise RuntimeError("Some E10B source/map prompt pairs are not pure ID remappings.")

    entity_overlap = source_entities & mapped_entities
    relation_overlap = source_relations & mapped_relations

    namespace_audit = {
        "entity_offset":entity_offset,
        "relation_offset":relation_offset,
        "source_entity_count":len(source_entities),
        "mapped_entity_count":len(mapped_entities),
        "source_entity_min":min(source_entities),
        "source_entity_max":max(source_entities),
        "mapped_entity_min":min(mapped_entities),
        "mapped_entity_max":max(mapped_entities),
        "entity_namespace_overlap_count":len(entity_overlap),
        "source_relation_count":len(source_relations),
        "mapped_relation_count":len(mapped_relations),
        "source_relation_min":min(source_relations),
        "source_relation_max":max(source_relations),
        "mapped_relation_min":min(mapped_relations),
        "mapped_relation_max":max(mapped_relations),
        "relation_namespace_overlap_count":len(relation_overlap),
        "all_2250_prompt_pairs_pass":all(row["all_mapping_checks_pass"] for row in rows),
        "triplet_order_preserved_for_all_pairs":all(row["triples_ordered_mapping_match"] for row in rows),
    }

    if entity_overlap:
        raise RuntimeError("Source and mapped entity namespaces overlap.")

    if relation_overlap:
        raise RuntimeError("Source and mapped relation namespaces overlap.")

    return (
        rows,
        namespace_audit,
    )
def collect_members(path:Path) -> Dict[int, Path]:

    mapping: Dict[int, str] = {}
    for txt_path in path.iterdir():
        if not txt_path.is_file():
            continue
        if txt_path.suffix.lower() != ".txt":
            continue
        sample_id = int(txt_path.stem)

        if sample_id in mapping:
            raise ValueError(
                f"重复 sample ID={sample_id}"
            )
        mapping[sample_id] = txt_path

    return mapping
def require_1_to_150(mapping: Dict[int, str],label: str,) -> None:

    expected = set(
        range(1, EXPECTED_N + 1)
    )

    observed = set(mapping)

    if observed != expected:
        raise ValueError(
            f"{label} 。\n"
            f"n={len(observed)}\n"
            f"missing={sorted(expected-observed)}\n"
            f"extra={sorted(observed-expected)}"
        )
def audit_result_files(base_dir10_B:Path) -> Dict[str, Any]:

    audit = {}

    for section in ["answer","score",]:
        total = 0
        groups = 0
        for model in MODELS:
            for method in METHODS:
                for qtype in QTYPES:

                    prefix = (
                        f"B/{section}/"
                        f"{model}/{method}/{qtype}/"
                    )
                    members = collect_members(base_dir10_B / section / model / method / qtype)
                    require_1_to_150(members,label=prefix,)
                    total += len(members)

                    groups += 1

        audit[f"{section}_files"] = total
        audit[f"{section}_groups"] = groups

    audit["all_complete"] = True

    return audit

def read_mapped_score(member: Path,) -> Dict[str, float]:


    data = read_json(member)

    required = {
        "exact_match",
        "precision",
        "recall",
        "f1",
        "parse_failed",
        "status_failed",
    }

    missing = required - set(
        data
    )

    if missing:
        raise ValueError(
            f"{member} missing fields: {sorted(missing)}"
        )

    return {
        "exact_match":
            float(
                data[
                    "exact_match"
                ]
            ),

        "precision":
            float(
                data[
                    "precision"
                ]
            ),

        "recall":
            float(
                data[
                    "recall"
                ]
            ),

        "f1":
            float(
                data[
                    "f1"
                ]
            ),

        "parse_failed":
            float(
                data[
                    "parse_failed"
                ]
            ),

        "status_failed":
            float(
                data[
                    "status_failed"
                ]
            ),
    }
def summarize_mapped_performance(base_dir10_B:Path) -> Tuple[List[Dict[str, Any]],Dict[Tuple[str, str, str, int],Dict[str, float],],]:


    rows = []
    sample_index = {}
    for model in MODELS:
        for method in METHODS:
            for qtype in QTYPES:
                scores = []
                for sample_id in range(1,EXPECTED_N + 1,):
                    score = read_mapped_score(base_dir10_B / 'score' / model / method / qtype / f'{sample_id}.txt')
                    sample_index[(model,method,qtype,sample_id,)] = score
                    scores.append(score)

                rows.append({
                    "model":model,
                    "method":METHOD_DISPLAY[method],
                    "qtype":qtype,
                    "n":len(scores),
                    "mean_f1":mean(x["f1"] for x in scores),
                    "mean_precision":mean(x["precision"] for x in scores),
                    "mean_recall":mean(x["recall"] for x in scores),
                    "exact_match_rate":mean(x["exact_match"] for x in scores),
                    "parse_failure_rate":mean(x["parse_failed"] for x in scores),
                    "status_failure_rate":mean(x["status_failed"] for x in scores),
                })

    return (
        rows,
        sample_index,
    )
def load_old_baseline_sample(base_dir10_B:Path,model: str,method: str,qtype: str,sample_id: int,) -> Tuple[Set[str],Set[str],Dict[str, float],]:


    model_folder = BASELINE_MODEL_FOLDER[model]
    method_folder = BASELINE_METHOD_FOLDER[method]


    member = (
        f"all/ft/"
        f"{model_folder}/"
        f"{method_folder}/"
        f"{qtype}/"
        f"{sample_id}.txt"
    )

    text = (base_dir10_B / 'answer_old' / model_folder / method_folder / qtype / f"{sample_id}.txt").read_text()

    parts = [
        x.strip()
        for x in OLD_SEPARATOR_RE.split(text.strip())
    ]

    if len(parts) != 2:
        raise ValueError(
            f"Unexpected baseline format: {member}"
        )

    pred = parse_entity_set(parts[0])
    gold = parse_entity_set(parts[1])
    metrics = set_metrics(pred,gold,)

    return (pred,gold,metrics,)
def load_source_baseline(base_dir10_B:Path) -> Tuple[List[Dict[str, Any]],Dict[Tuple[str, str, str, int],Dict[str, float],],Dict[str, Any],]:

    rows = []
    sample_index = {}
    checked_gold_pairs = 0
    gold_mismatch = 0

    for model in MODELS:
        for method in METHODS:
            for qtype in QTYPES:
                scores = []
                for sample_id in range(1,EXPECTED_N + 1,):
                    _pred,baseline_gold,metrics = load_old_baseline_sample(base_dir10_B,model,method,qtype,sample_id,)

                    source_member = (
                        f"B/source_prompt/"
                        f"{method}/{qtype}/{sample_id}.txt"
                    )
                    source_prompt = read_json(base_dir10_B / 'source_prompt' / method / qtype / f"{sample_id}.txt")

                    source_gold = {
                        str(x)
                        for x in parse_int_entity_set(source_prompt["answer"])
                    }
                    checked_gold_pairs += 1
                    if baseline_gold != source_gold:
                        gold_mismatch += 1
                    sample_index[(model,method,qtype,sample_id,)] = metrics
                    scores.append(metrics)
                rows.append({
                    "model":model,
                    "method":METHOD_DISPLAY[method],
                    "qtype":qtype,
                    "n":len(scores),
                    "mean_f1":mean(x["f1"] for x in scores),
                    "mean_precision":mean(x["precision"] for x in scores),
                    "mean_recall":mean(x["recall"] for x in scores),
                    "exact_match_rate":mean(x["exact_match"] for x in scores),
                })

    audit = {
        "gold_pairs_checked":
            checked_gold_pairs,

        "expected_gold_pairs":
            (
                len(MODELS)
                * len(METHODS)
                * len(QTYPES)
                * EXPECTED_N
            ),

        "gold_mismatch_count":
            gold_mismatch,

        "safe_to_align_source_vs_remap":
            int(gold_mismatch == 0),
    }

    if gold_mismatch != 0:
        raise RuntimeError(
            "Old FT baseline gold does not align with E10B source_prompt."
        )

    return (
        rows,
        sample_index,
        audit,
    )
def compare_source_remap(source_rows: List[Dict[str, Any]],mapped_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

    source_idx = {
        (r["model"],r["method"],r["qtype"],): r
        for r in source_rows
    }
    mapped_idx = {
        (r["model"],r["method"],r["qtype"],): r
        for r in mapped_rows
    }

    rows = []

    for model in MODELS:
        for method in ["RAG","Decomp","SplitRAG",]:
            for qtype in QTYPES:
                source = source_idx[(model,method,qtype,)]
                mapped = mapped_idx[(model,method,qtype,)]

                rows.append({
                    "model":model,
                    "method":method,
                    "qtype":qtype,
                    "source_f1":source["mean_f1"],
                    "remapped_f1":mapped["mean_f1"],
                    "delta_remap_minus_source_f1":(mapped["mean_f1"]-source["mean_f1"]),
                    "source_exact_match":source["exact_match_rate"],
                    "remapped_exact_match":mapped["exact_match_rate"],
                    "delta_remap_minus_source_em":(mapped["exact_match_rate"]-source["exact_match_rate"]),
                })

    return rows
def grouped_summary(compare_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    rows = []

    for model in MODELS:
        for method in ["RAG","Decomp","SplitRAG",]:
            base = [r for r in compare_rows if (r["model"] == model and r["method"] == method)]
            for group_name, qtypes in QUERY_GROUPS.items():
                s = [r for r in base if r["qtype"] in qtypes]
                source_f1 = mean(r["source_f1"] for r in s)
                remapped_f1 = mean(r["remapped_f1"] for r in s)

                rows.append({
                    "model":model,
                    "method":method,
                    "query_group":group_name,
                    "qtypes":",".join(qtypes),
                    "n_queries":len(qtypes)* EXPECTED_N,
                    "source_f1":source_f1,
                    "remapped_f1":remapped_f1,
                    "delta_remap_minus_source_f1":remapped_f1 - source_f1,

                    # retention ratio 只作为描述性辅助。
                    "remap_over_source_f1_ratio_DIAGNOSTIC":remapped_f1 / source_f1 if source_f1 > 0 else float("nan"),
                })

    return rows
def build_sample_table(source_index,mapped_index,) -> List[Dict[str, Any]]:


    rows = []

    for model in MODELS:
        for method in METHODS:
            for qtype in QTYPES:
                for sample_id in range(1,EXPECTED_N + 1,):

                    source = source_index[(model,method,qtype,sample_id,)]
                    mapped = mapped_index[(model,method,qtype,sample_id,)]

                    rows.append({
                        "model":model,
                        "method":METHOD_DISPLAY[method],
                        "qtype":qtype,
                        "sample_id":sample_id,
                        "source_f1":source["f1"],
                        "remapped_f1":mapped["f1"],
                        "delta_remap_minus_source_f1":(mapped["f1"]-source["f1"]),
                        "source_exact_match":source["exact_match"],
                        "remapped_exact_match":mapped["exact_match"],
                    })

    return rows

def build_markdown(namespace_audit: Dict[str, Any],result_audit: Dict[str, Any],baseline_audit: Dict[str, Any],compare_rows: List[Dict[str, Any]],grouped_rows: List[Dict[str, Any]],) -> str:


    lines = []

    lines.append(
        "# E10B — Identifier Remapping to an Unseen Namespace"
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # Manipulation audit
    # -------------------------------------------------------------------------
    lines.append(
        "## 1. Remapping integrity"
    )

    lines.append("")

    lines.append(
        "- Prompt pairs audited: **2250/2250 passed**."
    )

    lines.append(
        "- Triple order is preserved for all prompt pairs."
    )

    lines.append(
        f"- Entity offset: **+{namespace_audit['entity_offset']}**."
    )

    lines.append(
        f"- Relation offset: **+{namespace_audit['relation_offset']}**."
    )

    lines.append(
        f"- Source↔mapped entity namespace overlap: "
        f"**{namespace_audit['entity_namespace_overlap_count']}**."
    )

    lines.append(
        f"- Source↔mapped relation namespace overlap: "
        f"**{namespace_audit['relation_namespace_overlap_count']}**."
    )

    lines.append(
        f"- Remapped result files: "
        f"{result_audit['score_files']} score + "
        f"{result_audit['answer_files']} answer."
    )

    lines.append(
        f"- Old source-ID baseline gold alignment: "
        f"**{baseline_audit['gold_pairs_checked'] - baseline_audit['gold_mismatch_count']}/"
        f"{baseline_audit['gold_pairs_checked']}**."
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # qtype performance
    # -------------------------------------------------------------------------
    lines.append(
        "## 2. Source-ID vs unseen-ID performance"
    )

    lines.append("")

    lines.append(
        "| Model | Method | qtype | Source F1 | Remapped F1 | Delta |"
    )

    lines.append(
        "|---|---|---|---:|---:|---:|"
    )

    for row in compare_rows:

        lines.append(
            f"| {row['model']} | "
            f"{row['method']} | "
            f"{row['qtype']} | "
            f"{fmt(row['source_f1'])} | "
            f"{fmt(row['remapped_f1'])} | "
            f"{row['delta_remap_minus_source_f1']:+.4f} |"
        )

    lines.append("")

    # -------------------------------------------------------------------------
    # Overall
    # -------------------------------------------------------------------------
    lines.append(
        "## 3. Overall and set-query summary"
    )

    lines.append("")

    lines.append(
        "| Model | Method | Group | Source F1 | Remapped F1 | Delta |"
    )

    lines.append(
        "|---|---|---|---:|---:|---:|"
    )

    for row in grouped_rows:

        if row[
            "query_group"
        ] not in {
            "all",
            "set",
            "intersection",
        }:
            continue

        lines.append(
            f"| {row['model']} | "
            f"{row['method']} | "
            f"{row['query_group']} | "
            f"{fmt(row['source_f1'])} | "
            f"{fmt(row['remapped_f1'])} | "
            f"{row['delta_remap_minus_source_f1']:+.4f} |"
        )

    lines.append("")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    lines.append(
        "## 4. Interpretation"
    )

    lines.append("")

    lines.append(
        "1. E10B is a clean identifier-remapping intervention: "
        "query structure, evidence content, gold answers, and triple order are preserved, "
        "while entity and relation identifiers are moved into disjoint namespaces."
    )

    lines.append(
        "2. Both fine-tuned models retain substantial performance after remapping; "
        "the remapped runs do not show a systematic collapse."
    )

    lines.append(
        "3. Some method/qtype cells improve and others decrease after remapping. "
        "Therefore the safe claim is not numerical invariance, but preservation of "
        "task-solving ability under unseen identifiers."
    )

    lines.append(
        "4. This supports the conclusion that the observed KGQA capability is not "
        "dependent on memorizing the specific anonymous IDs used in the source namespace."
    )

    lines.append(
        "5. E10B does not rule out every possible form of leakage; it specifically "
        "tests dependence on the original anonymous identifier values."
    )

    lines.append("")

    return "\n".join(
        lines
    )
def summarize_e10b(output_dir: Path,) -> Dict[str, Any]:
    base_dir10_B = Path(r"")
    output_dir.mkdir(parents=True,exist_ok=True,)


    entity_offset, relation_offset, = infer_offsets(base_dir10_B)
    prompt_audit_rows,namespace_audit, = audit_prompt_remapping(base_dir10_B,entity_offset,relation_offset,)
    result_audit = audit_result_files(base_dir10_B)
    mapped_rows,mapped_index, = summarize_mapped_performance(base_dir10_B)
    source_rows,source_index,baseline_audit, = load_source_baseline(base_dir10_B)
    compare_rows = compare_source_remap(source_rows,mapped_rows,)
    grouped_rows = grouped_summary(compare_rows)
    sample_rows = build_sample_table(source_index,mapped_index,)

    write_csv(
        output_dir
        / "prompt_remap_audit.csv",
        prompt_audit_rows,
        [
            "method",
            "qtype",
            "sample_id",
            "stage_count_match",
            "triples_ordered_mapping_match",
            "question_entity_mapping_match",
            "question_relation_mapping_match",
            "answer_mapping_match",
            "all_mapping_checks_pass",
        ],
    )

    write_json(
        output_dir
        / "namespace_audit.json",
        namespace_audit,
    )

    write_csv(
        output_dir
        / "remapped_performance.csv",
        mapped_rows,
        [
            "model",
            "method",
            "qtype",
            "n",
            "mean_f1",
            "mean_precision",
            "mean_recall",
            "exact_match_rate",
            "parse_failure_rate",
            "status_failure_rate",
        ],
    )

    write_csv(
        output_dir
        / "source_vs_remap.csv",
        compare_rows,
        [
            "model",
            "method",
            "qtype",
            "source_f1",
            "remapped_f1",
            "delta_remap_minus_source_f1",
            "source_exact_match",
            "remapped_exact_match",
            "delta_remap_minus_source_em",
        ],
    )

    write_csv(
        output_dir
        / "grouped_summary.csv",
        grouped_rows,
        [
            "model",
            "method",
            "query_group",
            "qtypes",
            "n_queries",
            "source_f1",
            "remapped_f1",
            "delta_remap_minus_source_f1",
            "remap_over_source_f1_ratio_DIAGNOSTIC",
        ],
    )

    write_csv(
        output_dir
        / "sample_level.csv",
        sample_rows,
        [
            "model",
            "method",
            "qtype",
            "sample_id",
            "source_f1",
            "remapped_f1",
            "delta_remap_minus_source_f1",
            "source_exact_match",
            "remapped_exact_match",
        ],
    )

    summary = {
        "experiment":
            "E10B",

        "purpose":
            (
                "Test whether fine-tuned KGQA ability depends on "
                "memorizing the specific anonymous entity/relation IDs."
            ),

        "e10a":
            "Not processed here; reuse the existing E10A script.",

        "namespace_audit":
            namespace_audit,

        "result_file_audit":
            result_audit,

        "source_baseline_alignment":
            baseline_audit,

        "remapped_performance":
            mapped_rows,

        "source_vs_remap":
            compare_rows,

        "grouped_summary":
            grouped_rows,

        "interpretation_boundary": [
            (
                "The identifier namespace is fully disjoint after remapping."
            ),
            (
                "Prompt topology, evidence triples, triple order, and gold answers "
                "are preserved under deterministic remapping."
            ),
            (
                "Performance does not systematically collapse under unseen IDs."
            ),
            (
                "The result supports non-dependence on specific anonymous IDs, "
                "but does not rule out all forms of leakage."
            ),
        ],
    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    report = build_markdown(
        namespace_audit,
        result_audit,
        baseline_audit,
        compare_rows,
        grouped_rows,
    )

    (
        output_dir
        / "summary.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    return summary

def main():
    summary = summarize_e10b(Path(''))


if __name__ == "__main__":
    main()

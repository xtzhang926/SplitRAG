import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
RUNS = [
    "old",
    "new",
]
METHODS = [
    "Decompose",
    "SplitRag",
]
METHOD_DISPLAY = {
    "Decompose": "Decomp",
    "SplitRag": "SplitRAG",
}
QTYPES = [
    "2i",
    "3i",
]
EXPECTED_N = 150
EXPECTED_STAGE_COUNT = {
    "2i": 3,
    "3i": 4,
}
EXPECTED_PROJECTION_COUNT = {
    "2i": 2,
    "3i": 3,
}

LEGACY_SEPARATOR_RE = re.compile(r"@\s*@")
STRICT_ANSWER_RE = re.compile(
r"^(?:None|\d+(?:\s*,\s*\d+)*)$",flags=re.IGNORECASE,)

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
def mean(values: Iterable[float]) -> float:

    values = list(values)

    if not values:
        return float("nan")

    return sum(values) / len(values)
def fmt(value: float, digits: int = 4) -> str:

    if value is None:
        return "NA"

    if isinstance(value, float) and math.isnan(value):
        return "NA"

    return f"{value:.{digits}f}"
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
def audit_files(base_dir7) -> Dict[str, Any]:


    groups = []
    total_files = 0
    for run in RUNS:
        for source in ["gold",*METHODS,]:
            for qtype in QTYPES:
                prefix = (
                    f"{run}/"
                    f"{source}/"
                    f"{qtype}/"
                )
                members = collect_members(base_dir7 / f"Qwen3.5-0.8B_{run}" / source / qtype)
                require_1_to_150(members,label=prefix,)
                total_files += len(members)
                groups.append({
                    "run":run,
                    "source":source,
                    "qtype":qtype,
                    "n":len(members),
                    "ids_ok":True,
                })

    return {
        "total_txt_files":total_files,
        "expected_total_txt_files":len(RUNS) * (len(METHODS) + 1) * len(QTYPES) * EXPECTED_N,
        "groups":groups,
        "all_complete":True,
    }

def parse_answer_set(text: Any,) -> Optional[Set[int]]:


    if text is None:
        return None

    value = str(text).strip()

    if not value:

        return None

    if STRICT_ANSWER_RE.fullmatch(value) is None:
        return None

    if value.casefold() == "none":
        return set()

    return {
        int(x.strip())
        for x in value.split(",")
    }
def split_legacy_parts(text: str,) -> List[str]:

    return [
        part.strip()
        for part in LEGACY_SEPARATOR_RE.split(text.strip())
    ]
def set_metrics(pred: Optional[Set[int]],gold: Set[int],) -> Dict[str, float]:


    if pred is None:
        return {
            "exact_match": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    if not pred and not gold:
        return {
            "exact_match": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    if not pred or not gold:
        return {
            "exact_match":
                int(pred == gold),
            "precision":
                0.0,
            "recall":
                0.0,
            "f1":
                0.0,
        }

    inter = len(pred & gold)
    precision = inter / len(pred)
    recall = inter / len(gold)
    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "exact_match":
            int(pred == gold),
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
    }
def contains_gold(pred: Optional[Set[int]],gold: Set[int],) -> int:

    if pred is None:
        return 0
    return int(gold.issubset(pred))

def load_gold(base_dir7:Path,run: str,qtype: str,sample_id: int,) -> Tuple[List[Set[int]],Set[int],]:


    member = (
        f"E7/Qwen3.5-0.8B_{run}/"
        f"gold/{qtype}/{sample_id}.txt"
    )
    text = (base_dir7 / f"Qwen3.5-0.8B_{run}" / 'gold' / qtype / f"{sample_id}.txt").read_text()
    parts = split_legacy_parts(text)

    expected_stage_count = EXPECTED_STAGE_COUNT[qtype]
    expected_part_count = expected_stage_count + 1
    if len(parts) != expected_part_count:
        raise ValueError(
            f"Unexpected gold part count: {member}; "
            f"observed={len(parts)}, "
            f"expected={expected_part_count}"
        )
    parsed = []

    for part in parts:
        value = parse_answer_set(part)
        if value is None:
            raise ValueError(
                f"Gold parse failure: {member}: {part!r}"
            )
        parsed.append(value)

    stage_gold = parsed[:-1]
    final_gold = parsed[-1]

    # 最后 stage gold 是 final aggregation stage 的 gold。
    # 额外保存的 final_gold 在集合意义上应与它一致。
    if stage_gold[-1] != final_gold:
        raise ValueError(
            f"Final-stage gold != extra final gold as sets: {member}"
        )

    return (
        stage_gold,
        final_gold,
    )
def load_old_predictions(base_dir7:Path, method: str,qtype: str,sample_id: int,) -> Tuple[List[Optional[Set[int]]],Set[int],]:


    member = (
        f"E7/Qwen3.5-0.8B_old/"
        f"{method}/{qtype}/{sample_id}.txt"
    )
    text = (base_dir7 / f"Qwen3.5-0.8B_old" / method / qtype / f"{sample_id}.txt").read_text()
    parts = split_legacy_parts(text)
    expected_stage_count = EXPECTED_STAGE_COUNT[qtype]
    if len(parts) != (expected_stage_count + 1):
        raise ValueError(
            f"Unexpected old answer part count: {member}; "
            f"observed={len(parts)}, "
            f"expected={expected_stage_count+1}"
        )
    pred_parts = parts[:-1]
    appended_gold_text = parts[-1]
    pred = [parse_answer_set(x) for x in pred_parts]
    appended_gold = parse_answer_set(appended_gold_text)
    if appended_gold is None:
        raise ValueError(
            f"Old appended gold parse failure: {member}"
        )

    return (
        pred,
        appended_gold,
    )
def load_new_predictions(base_dir7:Path, method: str,qtype: str,sample_id: int,) -> Tuple[List[Optional[Set[int]]],Dict[str, Any],]:


    member = (
        f"E7/Qwen3.5-0.8B_new/"
        f"{method}/{qtype}/{sample_id}.txt"
    )
    record = read_json(base_dir7 / f"Qwen3.5-0.8B_new" / method / qtype / f"{sample_id}.txt")
    answers = record.get("answers")
    raw_answers = record.get("raw_answers")
    confidences = record.get("confidences")
    status = record.get("status")
    expected_stage_count = EXPECTED_STAGE_COUNT[qtype]
    if not isinstance(answers,list,):
        raise ValueError(
            f"Missing answers list: {member}"
        )
    if len(answers) != expected_stage_count:
        raise ValueError(
            f"Unexpected new stage count: {member}; "
            f"observed={len(answers)}, "
            f"expected={expected_stage_count}"
        )

    # raw_answers / confidences 应与 answers 对齐。
    if (
        not isinstance(raw_answers, list) or not isinstance(confidences, list)
        or len(raw_answers) != len(answers) or len(confidences) != len(answers)
    ):
        raise ValueError(
            f"New answer metadata length mismatch: {member}"
        )
    pred = [parse_answer_set(x) for x in answers]
    metadata = {
        "status":str(status),
        "confidences":[str(x) for x in confidences],
        "raw_answers":raw_answers,
        "normalized_answers":answers,
    }

    return (
        pred,
        metadata,
    )
def symbolic_intersection(projection_predictions: List[Optional[Set[int]]],) -> Optional[Set[int]]:

    if not projection_predictions:
        return None

    if any(x is None for x in projection_predictions):
        return None

    result = set(projection_predictions[0])
    for value in projection_predictions[1:]:
        result &= value

    return result
def analyze_one_sample(base_dir7:Path, run: str,method: str,qtype: str,sample_id: int,) -> Tuple[List[Dict[str, Any]],Dict[str, Any],]:

    stage_gold, final_gold = load_gold(base_dir7,run,qtype,sample_id,)
    if run == "old":
        pred_stages, appended_gold = load_old_predictions(base_dir7,method,qtype,sample_id,)

        if appended_gold != final_gold:
            raise ValueError(
                "Old answer appended gold mismatch: "
                f"{method}/{qtype}/{sample_id}"
            )
        new_status = None
        any_confidence_failed = None
    elif run == "new":
        pred_stages, metadata = load_new_predictions(base_dir7,method,qtype,sample_id,)
        new_status = metadata["status"]
        any_confidence_failed = int(any(x.casefold() == "failed" for x in metadata["confidences"]))
    else:
        raise ValueError(run)

    # -------------------------------------------------------------------------
    # stage 数量必须严格匹配 gold
    # -------------------------------------------------------------------------
    if len(pred_stages) != len(stage_gold):
        raise ValueError(
            f"Prediction/gold stage mismatch: "
            f"{run}/{method}/{qtype}/{sample_id}"
        )

    projection_count = EXPECTED_PROJECTION_COUNT[qtype]
    stage_rows = []

    # -------------------------------------------------------------------------
    # 每 stage metrics
    # -------------------------------------------------------------------------
    for stage_index, (pred,gold,) in enumerate(zip(pred_stages,stage_gold,),start=1,):
        metrics = set_metrics(pred,gold,)
        is_projection = int(stage_index <= projection_count)
        role = "projection" if is_projection else "final_aggregation"
        stage_rows.append({
            "run":run,
            "primary_result":int(run == "new"),
            "method":METHOD_DISPLAY[method],
            "qtype":qtype,
            "sample_id":sample_id,
            "stage":stage_index,
            "stage_role":role,
            "parse_ok":int(pred is not None),
            "pred_size":len(pred) if pred is not None else None,
            "gold_size":len(gold),
            "exact_match":metrics["exact_match"],
            "precision":metrics["precision"],
            "recall":metrics["recall"],
            "f1":metrics["f1"],
            "gold_contained":contains_gold(pred,gold,),
        })

    # -------------------------------------------------------------------------
    # Projection-level mechanism
    # -------------------------------------------------------------------------
    projection_pred = pred_stages[:projection_count]
    projection_gold = stage_gold[:projection_count]


    projection_exact_flags = [
        int(pred is not None and pred == gold)
        for pred, gold
        in zip(projection_pred,projection_gold,)
    ]

    projection_containment_flags = [
        contains_gold(pred,gold,)
        for pred, gold
        in zip(projection_pred,projection_gold,)
    ]
    all_projection_exact = int(all(projection_exact_flags))
    all_projection_contain = int(all(projection_containment_flags))

    # “可恢复但不完美”的核心 condition：
    # 所有 branch 都保留 gold，但至少有一个 branch 有额外/错误实体。
    recoverable_imperfect = int(all_projection_contain and not all_projection_exact)

    # -------------------------------------------------------------------------
    # Final LLM result
    # -------------------------------------------------------------------------
    final_pred = pred_stages[-1]
    final_metrics = set_metrics(final_pred,final_gold,)
    exact_recovery = int(recoverable_imperfect and final_metrics["exact_match"])

    # -------------------------------------------------------------------------
    # Symbolic diagnostic
    # -------------------------------------------------------------------------
    symbolic_pred = symbolic_intersection(projection_pred)
    symbolic_metrics = set_metrics(symbolic_pred,final_gold,)
    if final_metrics["f1"] > symbolic_metrics["f1"] + 1e-12:
        llm_vs_symbolic = "LLM_better"
    elif symbolic_metrics["f1"] > final_metrics["f1"] + 1e-12:
        llm_vs_symbolic = "Symbolic_better"
    else:
        llm_vs_symbolic = "Tie"
    sample_summary = {
        "run":run,
        "primary_result":int(run == "new"),
        "method":METHOD_DISPLAY[method],
        "qtype":qtype,
        "sample_id":sample_id,
        "all_projection_exact":all_projection_exact,
        "all_projection_contain_gold":all_projection_contain,
        "recoverable_imperfect_intermediate":recoverable_imperfect,
        "final_exact_match":final_metrics["exact_match"],
        "final_precision":final_metrics["precision"],
        "final_recall":final_metrics["recall"],
        "final_f1":final_metrics["f1"],
        "exact_recovery_from_imperfect_contained":exact_recovery,
        "symbolic_intersection_f1":symbolic_metrics["f1"],
        "symbolic_intersection_exact":symbolic_metrics["exact_match"],
        "llm_vs_symbolic":llm_vs_symbolic,
        "new_status":new_status,
        "any_confidence_failed":any_confidence_failed,
    }

    return (
        stage_rows,
        sample_summary,
    )
def analyze_all(base_dir7:Path) -> Tuple[List[Dict[str, Any]],List[Dict[str, Any]],]:


    all_stage_rows = []
    all_sample_rows = []

    for run in RUNS:
        for method in METHODS:
            for qtype in QTYPES:
                for sample_id in range(1,EXPECTED_N + 1,):
                    stage_rows, sample_row = analyze_one_sample(base_dir7,run,method,qtype,sample_id,)
                    all_stage_rows.extend(stage_rows)
                    all_sample_rows.append(sample_row)

    return (
        all_stage_rows,
        all_sample_rows,
    )

def summarize_stage_metrics(stage_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

    groups: Dict[Tuple[str,str,str,int,str,],List[Dict[str, Any]],] = {}

    for row in stage_rows:

        key = (
            row["run"],
            row["method"],
            row["qtype"],
            row["stage"],
            row["stage_role"],
        )

        groups.setdefault(key,[],).append(row)

    rows = []

    for key, s in groups.items():
        (run,method,qtype,stage,role,) = key

        rows.append({
            "run":run,
            "primary_result":int(run == "new"),
            "method":method,
            "qtype":qtype,
            "stage":stage,
            "stage_role":role,
            "n":len(s),
            "mean_f1":mean(x["f1"] for x in s),
            "mean_precision":mean(x["precision"] for x in s),
            "mean_recall":mean(x["recall"] for x in s),
            "exact_match_rate":mean(x["exact_match"] for x in s),
            "gold_containment_rate":mean(x["gold_contained"] for x in s),
            "parse_ok_rate":mean(x["parse_ok"] for x in s),
            "mean_pred_set_size":mean(x["pred_size"] for x in s if x["pred_size"] is not None),
            "mean_gold_set_size":mean(x["gold_size"] for x in s),
        })
    rows.sort(
        key=lambda r: (RUNS.index(r["run"]),["Decomp","SplitRAG",].index(r["method"]),QTYPES.index(r["qtype"]),r["stage"],)
    )
    return rows
def summarize_mechanism(stage_rows: List[Dict[str, Any]],sample_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    rows = []

    for run in RUNS:
        for method in ["Decomp","SplitRAG",]:
            for qtype in QTYPES:

                projections = [
                    r for r in stage_rows
                    if r["run"] == run
                       and r["method"] == method
                       and r["qtype"] == qtype
                       and r["stage_role"] == "projection"
                ]

                samples = [
                    r for r in sample_rows
                    if r["run"] == run
                       and r["method"] == method
                       and r["qtype"] == qtype
                ]

                recoverable = [
                    r for r in samples
                    if r["recoverable_imperfect_intermediate"] == 1
                ]

                all_contained = [
                    r for r in samples
                    if r["all_projection_contain_gold"] == 1
                ]

                # new JSON 额外状态诊断。
                if run == "new":
                    status_ok_rate = mean(
                        int(r["new_status"] == "ok")
                        for r in samples
                    )
                    confidence_failed_rate = mean(
                        int(r["any_confidence_failed"] or 0)
                        for r in samples
                    )
                else:
                    status_ok_rate = float("nan")
                    confidence_failed_rate = float("nan")

                rows.append({
                    "run":run,
                    "primary_result":int(run == "new"),
                    "method":method,
                    "qtype":qtype,
                    "n_queries":len(samples),
                    "n_projection_stage_observations":len(projections),
                    "projection_mean_f1":mean(r["f1"] for r in projections),
                    "projection_mean_precision":mean(r["precision"] for r in projections),
                    "projection_mean_recall":mean(r["recall"] for r in projections),
                    "projection_exact_match_rate":mean(r["exact_match"] for r in projections),
                    "projection_gold_containment_rate":mean(r["gold_contained"] for r in projections),
                    "all_projection_exact_rate":mean(r["all_projection_exact"] for r in samples),
                    "all_projection_containment_rate":mean(r["all_projection_contain_gold"] for r in samples),
                    "final_mean_f1":mean(r["final_f1"] for r in samples),
                    "final_mean_precision":mean(r["final_precision"] for r in samples),
                    "final_mean_recall":mean(r["final_recall"] for r in samples),
                    "final_exact_match_rate":mean(r["final_exact_match"] for r in samples),
                    "recoverable_imperfect_n":len(recoverable),
                    "exact_recovery_n":sum(r["exact_recovery_from_imperfect_contained"] for r in recoverable),
                    "exact_recovery_rate_given_recoverable":mean(r["final_exact_match"] for r in recoverable) if recoverable else float("nan"),
                    "mean_final_f1_given_recoverable":mean(r["final_f1"] for r in recoverable) if recoverable else float("nan"),
                    "all_contained_n":len(all_contained),
                    "final_exact_rate_given_all_contained":(mean(r["final_exact_match"] for r in all_contained) if all_contained else float("nan")),
                    "new_status_ok_rate":status_ok_rate,
                    "new_any_confidence_failed_rate":confidence_failed_rate,
                })

    return rows
def summarize_symbolic(sample_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

    rows = []

    for run in RUNS:
        for method in ["Decomp","SplitRAG",]:
            for qtype in QTYPES:
                s = [
                    r
                    for r in sample_rows
                    if (
                        r["run"] == run
                        and r["method"] == method
                        and r["qtype"] == qtype
                    )
                ]

                rows.append({
                    "run":run,
                    "primary_result":int(run == "new"),
                    "method":method,
                    "qtype":qtype,
                    "n":len(s),
                    "llm_final_mean_f1":mean(r["final_f1"] for r in s),
                    "symbolic_intersection_mean_f1_DIAGNOSTIC":mean(r["symbolic_intersection_f1"] for r in s),
                    "llm_minus_symbolic_f1_DIAGNOSTIC":mean(r["final_f1"] for r in s) - mean(r["symbolic_intersection_f1"]for r in s),
                    "llm_better_count":sum(r["llm_vs_symbolic"] == "LLM_better" for r in s),
                    "symbolic_better_count":sum(r["llm_vs_symbolic"] == "Symbolic_better" for r in s),
                    "tie_count":sum(r["llm_vs_symbolic"] == "Tie" for r in s),
                })

    return rows



def summarize_e7(output_dir: Path,) -> Dict[str, Any]:

    base_dir7 = Path(r"")


    output_dir.mkdir(parents=True,exist_ok=True,)




    file_audit = audit_files(base_dir7)

    stage_sample_rows, sample_rows = analyze_all(base_dir7)


    stage_metrics = summarize_stage_metrics(stage_sample_rows)
    mechanism_rows = summarize_mechanism(stage_sample_rows,sample_rows,)
    symbolic_rows = summarize_symbolic(sample_rows)


    write_csv(
        output_dir
        / "stage_metrics.csv",
        stage_metrics,
        [
            "run",
            "primary_result",
            "method",
            "qtype",
            "stage",
            "stage_role",
            "n",
            "mean_f1",
            "mean_precision",
            "mean_recall",
            "exact_match_rate",
            "gold_containment_rate",
            "parse_ok_rate",
            "mean_pred_set_size",
            "mean_gold_set_size",
        ],
    )

    write_csv(
        output_dir
        / "mechanism_summary.csv",
        mechanism_rows,
        [
            "run",
            "primary_result",
            "method",
            "qtype",
            "n_queries",
            "n_projection_stage_observations",
            "projection_mean_f1",
            "projection_mean_precision",
            "projection_mean_recall",
            "projection_exact_match_rate",
            "projection_gold_containment_rate",
            "all_projection_exact_rate",
            "all_projection_containment_rate",
            "final_mean_f1",
            "final_mean_precision",
            "final_mean_recall",
            "final_exact_match_rate",
            "recoverable_imperfect_n",
            "exact_recovery_n",
            "exact_recovery_rate_given_recoverable",
            "mean_final_f1_given_recoverable",
            "all_contained_n",
            "final_exact_rate_given_all_contained",
            "new_status_ok_rate",
            "new_any_confidence_failed_rate",
        ],
    )

    write_csv(
        output_dir
        / "symbolic_diagnostic.csv",
        symbolic_rows,
        [
            "run",
            "primary_result",
            "method",
            "qtype",
            "n",
            "llm_final_mean_f1",
            "symbolic_intersection_mean_f1_DIAGNOSTIC",
            "llm_minus_symbolic_f1_DIAGNOSTIC",
            "llm_better_count",
            "symbolic_better_count",
            "tie_count",
        ],
    )



    write_csv(
        output_dir
        / "sample_level.csv",
        sample_rows,
        [
            "run",
            "primary_result",
            "method",
            "qtype",
            "sample_id",
            "all_projection_exact",
            "all_projection_contain_gold",
            "recoverable_imperfect_intermediate",
            "final_exact_match",
            "final_precision",
            "final_recall",
            "final_f1",
            "exact_recovery_from_imperfect_contained",
            "symbolic_intersection_f1",
            "symbolic_intersection_exact",
            "llm_vs_symbolic",
            "new_status",
            "any_confidence_failed",
        ],
    )

    # =========================================================================
    # JSON
    # =========================================================================

    summary = {
        "experiment":
            "E7",

        "purpose":
            (
                "Explain SplitRAG gains through "
                "intermediate recall/containment and final recovery."
            ),

        "primary_run":
            "new",

        "legacy_run":
            "old",

        "file_audit":
            file_audit,

        "stage_metrics":
            stage_metrics,

        "mechanism_summary":
            mechanism_rows,

        "symbolic_diagnostic":
            symbolic_rows,

    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    return summary


def main():


    summary = summarize_e7(Path(r''))



if __name__ == "__main__":
    main()

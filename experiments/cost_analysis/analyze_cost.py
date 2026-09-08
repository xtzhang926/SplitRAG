

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Tuple

MODELS = [
    "Qwen3.5-0.8B",
    "Qwen3-4B",
]
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


    "complex": [
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
def validate_score(score: Dict[str, Any],member: str,) -> None:

    required = {
        "exact_match",
        "precision",
        "recall",
        "f1",
        "parse_failed",
        "status_failed",
    }

    missing = required - set(score)

    if missing:
        raise ValueError(
            f"{member} missing score fields: {sorted(missing)}"
        )
def read_score(member: str,) -> Dict[str, Any]:

    score = read_json(member)

    validate_score(score,member,)

    return {
        "exact_match":float(score["exact_match"]),
        "precision":float(score["precision"]),
        "recall":float(score["recall"]),
        "f1":float(score["f1"]),
        "parse_failed":float(score["parse_failed"]),
        "status_failed":float(score["status_failed"]),
    }

def read_efficiency(base_dir8:Path, model: str, method: str, qtype: str,) -> Dict[str, Any]:


    member = (
        f"E8/answer/"
        f"{model}/"
        f"{method}/"
        f"{qtype}/"
        f"_efficiency.json"
    )

    data = read_json(base_dir8 / 'answer' / model / method / qtype / f"_efficiency.json")

    required = {
        "model",
        "method",
        "qtype",
        "num_queries",
        "total_calls",
        "calls_per_query",
        "total_input_tokens",
        "input_tokens_per_query",
        "input_tokens_per_call",
        "total_output_tokens",
        "output_tokens_per_query",
        "total_tokens_per_query",
        "inference_seconds",
        "amortized_seconds_per_query",
        "queries_per_second",
        "num_batches",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(
            f"{member} missing efficiency fields: {sorted(missing)}"
        )
    if data["model"] != model or data["method"] != method or data["qtype"] != qtype:
        raise ValueError(
            f"Efficiency identity mismatch: {member}"
        )

    if int(data["num_queries"]) != EXPECTED_N:
        raise ValueError(
            f"Unexpected num_queries in {member}: "
            f"{data['num_queries']}"
        )

    return data
def summarize_accuracy(base_dir8:Path) -> List[Dict[str, Any]]:


    rows = []

    for model in MODELS:
        for method in METHODS:
            for qtype in QTYPES:

                prefix = (
                    f"E8/score/"
                    f"{model}/"
                    f"{method}/"
                    f"{qtype}/"
                )

                members = collect_members(base_dir8 / 'score' / model / method / qtype)
                require_1_to_150(members,label=prefix,)
                scores = [read_score(members[sample_id],) for sample_id in sorted(members)]

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

    return rows
def summarize_efficiency(base_dir8:Path) -> List[Dict[str, Any]]:


    rows = []

    for model in MODELS:
        for method in METHODS:
            for qtype in QTYPES:
                data = read_efficiency(base_dir8,model,method,qtype,)

                rows.append({
                    "model":model,
                    "method":METHOD_DISPLAY[method],
                    "qtype":qtype,
                    "num_queries":int(data["num_queries"]),
                    "total_calls":int(data["total_calls"]),
                    "calls_per_query":float(data["calls_per_query"]),
                    "total_input_tokens":int(data["total_input_tokens"]),
                    "input_tokens_per_query":float(data["input_tokens_per_query"]),
                    "input_tokens_per_call":float(data["input_tokens_per_call"]),
                    "total_output_tokens":int(data["total_output_tokens"]),
                    "output_tokens_per_query":float(data["output_tokens_per_query"]),
                    "total_tokens_per_query":float(data["total_tokens_per_query"]),
                    "inference_seconds":float(data["inference_seconds"]),
                    "amortized_seconds_per_query":float(data["amortized_seconds_per_query"]),
                    "queries_per_second":float(data["queries_per_second"]),
                    "num_batches":int(data["num_batches"]),
                })

    return rows
def audit_efficiency(base_dir8:Path, efficiency_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    eff_idx = {
        (r["model"],r["method"],r["qtype"],): r
        for r in efficiency_rows
    }
    raw_method = {
        "RAG": "Rag",
        "Decomp": "Decompose",
        "SplitRAG": "SplitRag",
    }

    rows = []

    for model in MODELS:
        for method_display in ["RAG","Decomp","SplitRAG",]:
            for qtype in QTYPES:
                method = raw_method[method_display]

                prefix = (
                    f"E8/answer/"
                    f"{model}/"
                    f"{method}/"
                    f"{qtype}/"
                )

                members = collect_members(base_dir8 / 'answer' / model / method / qtype)
                require_1_to_150(members,label=prefix,)

                sample_total_calls = 0
                sample_total_input_tokens = 0
                sample_total_output_tokens = 0
                sample_total_tokens = 0

                for sample_id in sorted(members):
                    record = read_json(members[sample_id])
                    required = {
                        "calls",
                        "total_input_tokens",
                        "total_output_tokens",
                        "total_tokens",
                    }
                    missing = required - set(record)
                    if missing:
                        raise ValueError(
                            f"{members[sample_id]} "
                            f"missing token fields: {sorted(missing)}"
                        )

                    sample_total_calls += int(record["calls"])
                    sample_total_input_tokens += int(record["total_input_tokens"])
                    sample_total_output_tokens += int(record["total_output_tokens"])
                    sample_total_tokens += int(record["total_tokens"])

                eff = eff_idx[(model,method_display,qtype,)]
                expected_total_tokens = eff["total_input_tokens"] + eff["total_output_tokens"]

                rows.append({
                    "model":model,
                    "method":method_display,
                    "qtype":qtype,
                    "n":len(members),
                    "sample_sum_calls":sample_total_calls,
                    "efficiency_total_calls":eff["total_calls"],
                    "calls_match":int(sample_total_calls== eff["total_calls"]),
                    "sample_sum_input_tokens":sample_total_input_tokens,
                    "efficiency_total_input_tokens":eff["total_input_tokens"],
                    "input_tokens_match":int(sample_total_input_tokens== eff["total_input_tokens"]),
                    "sample_sum_output_tokens":sample_total_output_tokens,
                    "efficiency_total_output_tokens":eff["total_output_tokens"],
                    "output_tokens_match":int(sample_total_output_tokens== eff["total_output_tokens"]),
                    "sample_sum_total_tokens":sample_total_tokens,
                    "efficiency_total_tokens_derived":expected_total_tokens,
                    "total_tokens_match":int(sample_total_tokens== expected_total_tokens),
                    "all_core_fields_match":int(sample_total_calls == eff["total_calls"]
                                                and sample_total_input_tokens == eff["total_input_tokens"]
                                                and sample_total_output_tokens== eff["total_output_tokens"]
                                                and sample_total_tokens == expected_total_tokens),
                })

    return rows
def join_accuracy_cost(accuracy_rows: List[Dict[str, Any]],efficiency_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    aidx = {
        (r["model"],r["method"],r["qtype"],): r
        for r in accuracy_rows
    }
    eidx = {
        (r["model"],r["method"],r["qtype"],): r
        for r in efficiency_rows
    }

    rows = []

    for model in MODELS:
        for method in ["RAG","Decomp","SplitRAG",]:
            for qtype in QTYPES:
                a = aidx[(model,method,qtype,)]
                e = eidx[(model,method,qtype,)]
                rows.append({
                    "model":model,
                    "method":method,
                    "qtype":qtype,
                    "mean_f1":a["mean_f1"],
                    "exact_match_rate":a["exact_match_rate"],
                    "calls_per_query":e["calls_per_query"],
                    "input_tokens_per_query":e["input_tokens_per_query"],
                    "input_tokens_per_call":e["input_tokens_per_call"],
                    "output_tokens_per_query":e["output_tokens_per_query"],
                    "total_tokens_per_query":e["total_tokens_per_query"],
                    "amortized_seconds_per_query":e["amortized_seconds_per_query"],
                    "queries_per_second":e["queries_per_second"],


                    "f1_per_1k_input_tokens_DIAGNOSTIC":
                        (
                            1000.0
                            * a["mean_f1"]
                            / e["input_tokens_per_query"]
                            if e["input_tokens_per_query"] > 0
                            else float("nan")
                        ),
                })

    return rows
def grouped_summary(joined_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    rows = []

    for model in MODELS:
        for method in ["RAG","Decomp","SplitRAG",]:
            base = [r for r in joined_rows if r["model"] == model and r["method"] == method]

            for group_name, qtypes in QUERY_GROUPS.items():
                s = [r for r in base if r["qtype"] in qtypes]
                n_queries = len(s) * EXPECTED_N
                rows.append({
                    "model":model,
                    "method":method,
                    "query_group":group_name,
                    "qtypes":",".join(qtypes),
                    "n_queries":n_queries,
                    "mean_f1":mean(r["mean_f1"] for r in s),
                    "calls_per_query":mean(r["calls_per_query"] for r in s),
                    "input_tokens_per_query":mean(r["input_tokens_per_query"] for r in s),
                    "input_tokens_per_call":sum(r["input_tokens_per_query"] for r in s) / sum(r["calls_per_query"] for r in s),
                    "output_tokens_per_query":mean(r["output_tokens_per_query"] for r in s),
                    "total_tokens_per_query":mean(r["total_tokens_per_query"] for r in s),

                    "amortized_seconds_per_query":mean(r["amortized_seconds_per_query"] for r in s),
                })

    return rows
def split_vs_decomp(joined_rows: List[Dict[str, Any]],grouped_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    rows = []
    idx = {
        (r["model"],r["method"],r["qtype"],): r
        for r in joined_rows
    }

    # -------------------------------------------------------------------------
    # qtype-level
    # -------------------------------------------------------------------------
    for model in MODELS:
        for qtype in QTYPES:
            d = idx[(model,"Decomp",qtype,)]
            s = idx[(model,"SplitRAG",qtype,)]
            same_calls = int(abs(d["calls_per_query"] - s["calls_per_query"])< 1e-12)
            input_reduction = 1.0 - s["input_tokens_per_query"] / d["input_tokens_per_query"]
            total_reduction = 1.0 - s["total_tokens_per_query"] / d["total_tokens_per_query"]

            rows.append({
                "model":model,
                "scope_type":"qtype",
                "scope":qtype,
                "decomp_calls_per_query":d["calls_per_query"],
                "splitrag_calls_per_query":s["calls_per_query"],
                "same_calls":same_calls,

                "decomp_input_tokens_per_query":d["input_tokens_per_query"],
                "splitrag_input_tokens_per_query":s["input_tokens_per_query"],
                "input_token_reduction_fraction":input_reduction,
                "input_token_reduction_percent":100.0 * input_reduction,

                "decomp_total_tokens_per_query":d["total_tokens_per_query"],
                "splitrag_total_tokens_per_query":s["total_tokens_per_query"],
                "total_token_reduction_fraction":total_reduction,
                "total_token_reduction_percent":100.0 * total_reduction,

                "decomp_f1":d["mean_f1"],
                "splitrag_f1":s["mean_f1"],
                "delta_f1_split_minus_decomp":s["mean_f1"]- d["mean_f1"],
            })

    # -------------------------------------------------------------------------
    # query-group level
    # -------------------------------------------------------------------------
    gidx = {
        (r["model"],r["method"],r["query_group"],): r
        for r in grouped_rows
    }

    for model in MODELS:
        for group_name in QUERY_GROUPS:
            d = gidx[(model,"Decomp",group_name,)]
            s = gidx[(model,"SplitRAG",group_name,)]
            same_calls = int(abs(d["calls_per_query"]- s["calls_per_query"])< 1e-12)
            input_reduction = 1.0- s["input_tokens_per_query"]/ d["input_tokens_per_query"]
            total_reduction = 1.0- s["total_tokens_per_query"]/ d["total_tokens_per_query"]

            rows.append({
                "model":model,
                "scope_type":"group",
                "scope":group_name,

                "decomp_calls_per_query":d["calls_per_query"],
                "splitrag_calls_per_query":s["calls_per_query"],
                "same_calls":same_calls,

                "decomp_input_tokens_per_query":d["input_tokens_per_query"],
                "splitrag_input_tokens_per_query":s["input_tokens_per_query"],
                "input_token_reduction_fraction":input_reduction,
                "input_token_reduction_percent":100.0 * input_reduction,

                "decomp_total_tokens_per_query":d["total_tokens_per_query"],
                "splitrag_total_tokens_per_query":s["total_tokens_per_query"],
                "total_token_reduction_fraction":total_reduction,
                "total_token_reduction_percent":100.0 * total_reduction,

                "decomp_f1":d["mean_f1"],
                "splitrag_f1":s["mean_f1"],
                "delta_f1_split_minus_decomp":s["mean_f1"]- d["mean_f1"],
            })

    return rows


def summarize_e8(output_dir: Path,) -> Dict[str, Any]:

    base_dir8 = Path(r"")


    output_dir.mkdir(parents=True,exist_ok=True,)



    accuracy_rows = summarize_accuracy(base_dir8)
    efficiency_rows = summarize_efficiency(base_dir8)
    audit_rows = audit_efficiency(base_dir8,efficiency_rows)
    bad_audit = [r for r in audit_rows if not r["all_core_fields_match"]]

    if bad_audit:
        raise RuntimeError(
            "Efficiency summary does not match sample-level answer data. "
            "See audit output."
        )

    joined_rows = join_accuracy_cost(accuracy_rows,efficiency_rows,)
    grouped_rows = grouped_summary(joined_rows)
    comparison_rows = split_vs_decomp(joined_rows,grouped_rows,)

    # =========================================================================
    # CSV
    # =========================================================================

    write_csv(
        output_dir
        / "accuracy_by_qtype.csv",
        accuracy_rows,
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
        / "efficiency_by_qtype.csv",
        efficiency_rows,
        [
            "model",
            "method",
            "qtype",
            "num_queries",
            "total_calls",
            "calls_per_query",
            "total_input_tokens",
            "input_tokens_per_query",
            "input_tokens_per_call",
            "total_output_tokens",
            "output_tokens_per_query",
            "total_tokens_per_query",
            "inference_seconds",
            "amortized_seconds_per_query",
            "queries_per_second",
            "num_batches",
        ],
    )

    write_csv(
        output_dir
        / "accuracy_cost_joined.csv",
        joined_rows,
        [
            "model",
            "method",
            "qtype",
            "mean_f1",
            "exact_match_rate",
            "calls_per_query",
            "input_tokens_per_query",
            "input_tokens_per_call",
            "output_tokens_per_query",
            "total_tokens_per_query",
            "amortized_seconds_per_query",
            "queries_per_second",
            "f1_per_1k_input_tokens_DIAGNOSTIC",
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
            "mean_f1",
            "calls_per_query",
            "input_tokens_per_query",
            "input_tokens_per_call",
            "output_tokens_per_query",
            "total_tokens_per_query",
            "amortized_seconds_per_query",
        ],
    )

    write_csv(
        output_dir
        / "split_vs_decomp.csv",
        comparison_rows,
        [
            "model",
            "scope_type",
            "scope",
            "decomp_calls_per_query",
            "splitrag_calls_per_query",
            "same_calls",
            "decomp_input_tokens_per_query",
            "splitrag_input_tokens_per_query",
            "input_token_reduction_fraction",
            "input_token_reduction_percent",
            "decomp_total_tokens_per_query",
            "splitrag_total_tokens_per_query",
            "total_token_reduction_fraction",
            "total_token_reduction_percent",
            "decomp_f1",
            "splitrag_f1",
            "delta_f1_split_minus_decomp",
        ],
    )

    write_csv(
        output_dir
        / "efficiency_audit.csv",
        audit_rows,
        [
            "model",
            "method",
            "qtype",
            "n",
            "sample_sum_calls",
            "efficiency_total_calls",
            "calls_match",
            "sample_sum_input_tokens",
            "efficiency_total_input_tokens",
            "input_tokens_match",
            "sample_sum_output_tokens",
            "efficiency_total_output_tokens",
            "output_tokens_match",
            "sample_sum_total_tokens",
            "efficiency_total_tokens_derived",
            "total_tokens_match",
            "all_core_fields_match",
        ],
    )

    # =========================================================================
    # JSON
    # =========================================================================

    summary = {
        "experiment":
            "E8",

        "purpose":
            (
                "Characterize absolute inference cost and the "
                "accuracy-cost trade-off of staged evidence allocation."
            ),

        "integrity": {
            "models":
                MODELS,
            "methods":
                [
                    "RAG",
                    "Decomp",
                    "SplitRAG",
                ],
            "qtypes":
                QTYPES,
            "queries_per_group":
                EXPECTED_N,
            "score_records":
                (
                    len(MODELS)
                    * len(METHODS)
                    * len(QTYPES)
                    * EXPECTED_N
                ),
            "efficiency_groups":
                len(efficiency_rows),
            "efficiency_audit_all_match":
                True,
        },

        "accuracy_by_qtype":
            accuracy_rows,

        "efficiency_by_qtype":
            efficiency_rows,

        "grouped_summary":
            grouped_rows,

        "split_vs_decomp":
            comparison_rows,

        "interpretation_rules": [
            (
                "RAG has the lowest absolute call/token cost."
            ),
            (
                "Do not claim SplitRAG is cheaper than RAG."
            ),
            (
                "Decomp and SplitRAG have equal staged call counts, "
                "so their input-token difference measures context exposure "
                "under the same staged-call budget."
            ),
            (
                "For resource-limited 0.8B set/intersection queries, "
                "SplitRAG provides a better accuracy-cost trade-off."
            ),
            (
                "For stronger models near ceiling, selective deployment "
                "is more appropriate."
            ),
            (
                "Timing and QPS are descriptive because batching/output length differ."
            ),
        ],
    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )


    return summary


def main():


    summary = summarize_e8(Path(r''))




if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

MODELS = [
    "Llama-3.2-1B-Instruct",
    "Qwen3-4B",
    "Qwen3.5-0.8B",
    "Qwen3.5-4B",
]
QTYPES = [
    "1p",
    "2p",
    "2i",
    "3i",
    "2u",
]
METHOD_FOLDER = {
    "RAG": "abstract",
    "Decomp": "step",
    "SplitRAG": "step_rag",
    "CoT": "CoT_p1",
    "FS-CoT": "FewShot_CoT_p1",
}
METHODS = list(METHOD_FOLDER.keys())
COMPARISONS = [
    ("RAG", "Decomp", "RAG -> Decomp"),
    ("Decomp", "SplitRAG", "Decomp -> SplitRAG"),
    ("RAG", "SplitRAG", "RAG -> SplitRAG"),
    ("RAG", "CoT", "RAG -> CoT"),
    ("CoT", "FS-CoT", "CoT -> FS-CoT"),
    ("SplitRAG", "FS-CoT", "SplitRAG -> FS-CoT"),
]
EXPECTED_N = 150
BONFERRONI_M = 6
ALPHA = 0.05
OLD_SEPARATOR_RE = re.compile(r"@\s*@")
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
def parse_entity_set(text: str) -> Set[str]:


    value = str(text).strip()

    if not value:
        return set()

    if value.casefold() == "none":
        return set()

    result = set()

    for item in value.split(","):

        item = item.strip()

        if not item:
            continue

        if item.casefold() == "none":
            continue

        result.add(item)

    return result
def parse_old_result_text(text: str,member: str,) -> Tuple[Set[str], Set[str]]:

    parts = [x.strip() for x in OLD_SEPARATOR_RE.split(text.strip())]

    if len(parts) != 2:
        raise ValueError(
            f"Unexpected old result format: {member}; "
            f"parts={len(parts)}"
        )
    pred = parse_entity_set(parts[0])
    gold = parse_entity_set(parts[1])
    return pred, gold
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
def collect_formal_records(base_dir9:Path) -> Tuple[List[Dict[str, Any]],Dict[str, Any],]:


    rows = []
    group_audit = []
    # 用来审计同 sample ID 的 gold 是否一致。
    gold_index: Dict[Tuple[str, str, int],Dict[str, Set[str]]] = defaultdict(dict)
    for model in MODELS:
        for method in METHODS:
            folder = METHOD_FOLDER[method]
            for qtype in QTYPES:
                expected_ids = set(range(1,EXPECTED_N + 1,))
                observed_ids = set()

                for sample_id in range(1,EXPECTED_N + 1,):

                    member = (f"all/{model}/"
                        f"{folder}/"
                        f"{qtype}/"
                        f"{sample_id}.txt")

                    if not (base_dir9 / model / folder / qtype / f"{sample_id}.txt").exists():
                        raise FileNotFoundError(
                            f"Missing formal E9 file: {member}"
                        )

                    text = (base_dir9 / model / folder / qtype / f"{sample_id}.txt").read_text()
                    pred, gold = parse_old_result_text(text,member,)
                    metrics = set_metrics(pred,gold,)
                    observed_ids.add(sample_id)

                    gold_index[(model,qtype,sample_id,)][method] = gold

                    rows.append({
                        "model":model,
                        "method":method,
                        "source_folder":folder,
                        "qtype":qtype,
                        "sample_id":sample_id,
                        "exact_match":metrics["exact_match"],
                        "precision":metrics["precision"],
                        "recall":metrics["recall"],
                        "f1":metrics["f1"],
                        "pred_size":len(pred),
                        "gold_size":len(gold),
                    })

                if observed_ids != expected_ids:
                    raise ValueError(
                        f"Sample IDs mismatch: "
                        f"{model}/{method}/{qtype}"
                    )

                group_audit.append({
                    "model":model,
                    "method":method,
                    "qtype":qtype,
                    "n":len(observed_ids),
                    "ids_1_to_150":True,
                })

    # -------------------------------------------------------------------------
    # Gold 一致性
    # -------------------------------------------------------------------------
    gold_mismatches = []

    expected_method_set = set(METHODS)

    for key, method_gold in gold_index.items():

        if set(method_gold) != expected_method_set:

            gold_mismatches.append({
                "type":
                    "missing_method_gold",
                "key":
                    list(key),
                "methods":
                    sorted(
                        method_gold
                    ),
            })

            continue

        unique_gold = {
            tuple(sorted(value))
            for value in method_gold.values()
        }

        if len(unique_gold) != 1:

            gold_mismatches.append({
                "type":
                    "gold_mismatch",
                "model":
                    key[0],
                "qtype":
                    key[1],
                "sample_id":
                    key[2],
                "gold_by_method": {
                    method:
                        sorted(gold)
                    for method, gold
                    in method_gold.items()
                },
            })

    if gold_mismatches:
        raise RuntimeError(
            "Gold mismatch detected. "
            "Paired E9 analysis is invalid until fixed."
        )

    audit = {
        "formal_record_count":
            len(rows),

        "expected_formal_record_count":
            (
                len(MODELS)
                * len(METHODS)
                * len(QTYPES)
                * EXPECTED_N
            ),

        "group_count":
            len(group_audit),

        "all_groups_n150":
            all(
                x["n"] == EXPECTED_N
                for x in group_audit
            ),

        "gold_alignment_keys":
            len(gold_index),

        "expected_gold_alignment_keys":
            (
                len(MODELS)
                * len(QTYPES)
                * EXPECTED_N
            ),

        "gold_mismatch_count":
            len(gold_mismatches),

        "strict_sample_id_pairing_valid":
            True,

        "groups":
            group_audit,
    }

    return rows, audit
def summarize_methods(sample_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    groups = defaultdict(list)

    for row in sample_rows:
        groups[(row["model"],row["qtype"],row["method"],)].append(row)

    rows = []
    for model in MODELS:
        for qtype in QTYPES:
            for method in METHODS:
                s = groups[(model,qtype,method,)]

                if len(s) != EXPECTED_N:
                    raise ValueError(
                        f"Unexpected group n: "
                        f"{model}/{qtype}/{method}"
                    )

                rows.append({
                    "model":model,
                    "qtype":qtype,
                    "method":method,
                    "n":len(s),
                    "mean_f1":mean(x["f1"] for x in s),
                    "mean_precision":mean(x["precision"] for x in s),
                    "mean_recall":mean(x["recall"] for x in s),
                    "exact_match_rate":mean(x["exact_match"] for x in s),
                })

    return rows
def delta_ci(diff: np.ndarray,alpha: float = 0.05,) -> Tuple[float,float,float,]:


    diff = np.asarray(diff,dtype=float,)
    n = len(diff)
    mu = float(np.mean(diff))

    if n < 2:
        return (
            mu,
            float("nan"),
            float("nan"),
        )

    sd = float(np.std(diff,ddof=1,))

    if sd == 0:

        return (
            mu,
            mu,
            mu,
        )

    se = (
        sd
        / math.sqrt(n)
    )

    tcrit = float(
        stats.t.ppf(
            1 - alpha / 2,
            df=n - 1,
        )
    )

    return (
        mu,
        mu - tcrit * se,
        mu + tcrit * se,
    )
def cohen_dz(diff: np.ndarray,) -> float:

    diff = np.asarray(
        diff,
        dtype=float,
    )

    if len(diff) < 2:
        return float("nan")

    mu = float(
        np.mean(diff)
    )

    sd = float(
        np.std(
            diff,
            ddof=1,
        )
    )

    if sd == 0:

        if mu == 0:
            return 0.0

        return math.copysign(
            float("inf"),
            mu,
        )

    return mu / sd
def paired_t_test(a: np.ndarray,b: np.ndarray,) -> Tuple[float,float,]:


    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if np.array_equal(
        a,
        b,
    ):
        return 0.0, 1.0

    result = stats.ttest_rel(
        b,
        a,
        nan_policy="raise",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )
def wilcoxon_test(a: np.ndarray,b: np.ndarray,) -> Tuple[float,float,]:


    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if np.array_equal(
        a,
        b,
    ):
        return 0.0, 1.0

    try:

        result = stats.wilcoxon(
            b,
            a,
            zero_method="pratt",
            alternative="two-sided",
            method="auto",
        )

        return (
            float(result.statistic),
            float(result.pvalue),
        )

    except ValueError:

        return (
            float("nan"),
            float("nan"),
        )
def bonferroni_family6(p: float,) -> float:

    if p is None:
        return float("nan")

    if isinstance(p, float) and math.isnan(p):
        return float("nan")

    return min(
        1.0,
        float(p)
        * BONFERRONI_M,
    )
def paired_tests(sample_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    index = {}

    id_sets = defaultdict(set)

    for row in sample_rows:

        key = (row["model"],row["qtype"],row["method"],row["sample_id"],)
        if key in index:
            raise ValueError(
                f"Duplicate formal sample: {key}"
            )
        index[key] = row
        id_sets[(row["model"],row["qtype"],row["method"],)].add(row["sample_id"])

    expected_ids = set(range(1,EXPECTED_N + 1,))
    rows = []
    for model in MODELS:
        for qtype in QTYPES:
            # -----------------------------------------------------------------
            # 同一个 family 内的所有 5 个方法必须严格同 sample IDs。
            # -----------------------------------------------------------------
            for method in METHODS:

                ids = id_sets[(model,qtype,method,)]
                if ids != expected_ids:
                    raise ValueError(
                        f"Paired IDs invalid: "
                        f"{model}/{qtype}/{method}"
                    )

            for (method_a,method_b,label,) in COMPARISONS:
                ids = sorted(expected_ids)

                a = np.array(
                    [index[(model,qtype,method_a,sid,)]["f1"] for sid in ids],
                    dtype=float,
                )
                b = np.array(
                    [index[(model,qtype,method_b,sid,)]["f1"] for sid in ids],
                    dtype=float,
                )
                diff = b - a
                delta,ci_low,ci_high, = delta_ci(diff)
                t_stat,t_p, = paired_t_test(a,b,)
                w_stat,w_p, = wilcoxon_test(a,b,)
                dz = cohen_dz(diff)
                eps = 1e-12

                wins = int(np.sum(diff > eps))
                ties = int(np.sum(np.abs(diff)<= eps))
                losses = int(np.sum(diff < -eps))

                t_p_bonf = bonferroni_family6(t_p)
                w_p_bonf = bonferroni_family6(w_p)


                rows.append({
                    "model":model,
                    "qtype":qtype,
                    "comparison":label,
                    "method_A":method_a,
                    "method_B":method_b,
                    "delta_direction":"B-A",
                    "n":len(ids),
                    "mean_f1_A":float(np.mean(a)),
                    "mean_f1_B":float(np.mean(b)),
                    "delta_f1_B_minus_A":delta,
                    "delta_95ci_low":ci_low,
                    "delta_95ci_high":ci_high,
                    "cohen_dz":dz,
                    "paired_t_stat":t_stat,
                    "paired_t_p_raw":t_p,
                    "paired_t_p_bonferroni_family6":t_p_bonf,
                    "paired_t_sig_bonf_0_05":int(t_p_bonf< ALPHA),
                    "wilcoxon_stat":w_stat,
                    "wilcoxon_p_raw":w_p,
                    "wilcoxon_p_bonferroni_family6":w_p_bonf,
                    "wilcoxon_sig_bonf_0_05":int((not math.isnan(w_p_bonf)) and w_p_bonf< ALPHA),
                    "wins_B_gt_A":wins,
                    "ties":ties,
                    "losses_B_lt_A":losses,
                    "significant_direction":
                        (
                            "B_better" if t_p_bonf < ALPHA and delta > eps else
                            "B_worse" if t_p_bonf < ALPHA and delta < -eps else "not_significant"
                        ),
                })

    if len(rows) != 120:
        raise RuntimeError(
            f"Expected exactly 120 planned tests, got {len(rows)}."
        )

    return rows
def significance_summary(test_rows: List[Dict[str, Any]],) -> List[Dict[str, Any]]:


    rows = []

    def pack(scope_type: str,scope: str,subset: List[Dict[str, Any]],) -> Dict[str, Any]:

        t_sig = sum(
            r[
                "paired_t_sig_bonf_0_05"
            ]
            for r in subset
        )

        w_sig = sum(
            r[
                "wilcoxon_sig_bonf_0_05"
            ]
            for r in subset
        )

        favorable = sum(
            int(
                r[
                    "significant_direction"
                ] == "B_better"
            )
            for r in subset
        )

        unfavorable = sum(
            int(
                r[
                    "significant_direction"
                ] == "B_worse"
            )
            for r in subset
        )

        return {
            "scope_type":
                scope_type,

            "scope":
                scope,

            "n_tests":
                len(subset),

            "paired_t_significant_after_family_bonf":
                t_sig,

            "paired_t_significant_rate":
                (
                    t_sig
                    / len(subset)
                    if subset
                    else float("nan")
                ),

            "paired_t_significant_B_better":
                favorable,

            "paired_t_significant_B_worse":
                unfavorable,

            "wilcoxon_significant_after_family_bonf":
                w_sig,

            "wilcoxon_significant_rate":
                (
                    w_sig
                    / len(subset)
                    if subset
                    else float("nan")
                ),
        }

    rows.append(
        pack(
            "overall",
            "all_120_planned_tests",
            test_rows,
        )
    )

    for _, _, label in COMPARISONS:

        subset = [
            r
            for r in test_rows
            if r["comparison"] == label
        ]

        rows.append(
            pack(
                "comparison",
                label,
                subset,
            )
        )

    for model in MODELS:

        subset = [
            r
            for r in test_rows
            if r["model"] == model
        ]

        rows.append(
            pack(
                "model",
                model,
                subset,
            )
        )

    for qtype in QTYPES:

        subset = [
            r
            for r in test_rows
            if r["qtype"] == qtype
        ]

        rows.append(
            pack(
                "qtype",
                qtype,
                subset,
            )
        )

    return rows
def build_markdown(formal_audit: Dict[str, Any],method_rows: List[Dict[str, Any]],test_rows: List[Dict[str, Any]],significance_rows: List[Dict[str, Any]],) -> str:


    lines = []

    lines.append(
        "# E9 — Statistical Significance / Reliability"
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------
    lines.append(
        "## 1. Integrity"
    )
    lines.append("")

    lines.append(
        f"- Formal sample-level records: "
        f"**{formal_audit['formal_record_count']} / "
        f"{formal_audit['expected_formal_record_count']}**."
    )

    lines.append(
        f"- Gold alignment mismatches: "
        f"**{formal_audit['gold_mismatch_count']}**."
    )

    lines.append(
        "- Every formal method uses sample IDs **1..150** "
        "within each model×qtype."
    )

    lines.append(
        "- Extra FT/API/common results in the archive are intentionally "
        "excluded from the frozen E9 significance family."
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # Main significance result
    # -------------------------------------------------------------------------
    overall = next(
        r
        for r in significance_rows
        if r["scope_type"] == "overall"
    )

    lines.append(
        "## 2. Main result"
    )
    lines.append("")

    lines.append(
        f"- Planned paired t-tests: **{overall['n_tests']}**."
    )

    lines.append(
        f"- Significant after within-model×qtype Bonferroni correction "
        f"(6 planned contrasts per family): "
        f"**{overall['paired_t_significant_after_family_bonf']}/"
        f"{overall['n_tests']}**."
    )

    lines.append(
        f"- Wilcoxon robustness check significant after the same correction: "
        f"**{overall['wilcoxon_significant_after_family_bonf']}/"
        f"{overall['n_tests']}**."
    )

    lines.append("")

    lines.append(
        "Primary correction: "
        "`p_bonf = min(6 × p_raw, 1)`, "
        "equivalent to alpha = 0.05/6 ≈ 0.00833."
    )

    lines.append("")

    # -------------------------------------------------------------------------
    # By comparison
    # -------------------------------------------------------------------------
    lines.append(
        "## 3. Significant primary tests by planned comparison"
    )
    lines.append("")

    lines.append(
        "| Comparison | Significant / 20 | B significantly better | "
        "B significantly worse |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    for _, _, label in COMPARISONS:

        row = next(
            r
            for r in significance_rows
            if (
                r["scope_type"] == "comparison"
                and r["scope"] == label
            )
        )

        lines.append(
            f"| {label} | "
            f"{row['paired_t_significant_after_family_bonf']}/"
            f"{row['n_tests']} | "
            f"{row['paired_t_significant_B_better']} | "
            f"{row['paired_t_significant_B_worse']} |"
        )

    lines.append("")

    # -------------------------------------------------------------------------
    # 0.8B core results
    # -------------------------------------------------------------------------
    lines.append(
        "## 4. Qwen3.5-0.8B — core SplitRAG comparisons"
    )
    lines.append("")

    lines.append(
        "| qtype | Comparison | Delta F1 | 95% CI | dz | Bonferroni p |"
    )

    lines.append(
        "|---|---|---:|---|---:|---:|"
    )

    for qtype in QTYPES:

        for label in [
            "Decomp -> SplitRAG",
            "RAG -> SplitRAG",
        ]:

            row = next(
                r
                for r in test_rows
                if (
                    r["model"] == "Qwen3.5-0.8B"
                    and r["qtype"] == qtype
                    and r["comparison"] == label
                )
            )

            lines.append(
                f"| {qtype} | "
                f"{label} | "
                f"{row['delta_f1_B_minus_A']:+.4f} | "
                f"[{row['delta_95ci_low']:+.4f}, "
                f"{row['delta_95ci_high']:+.4f}] | "
                f"{fmt(row['cohen_dz'],3)} | "
                f"{row['paired_t_p_bonferroni_family6']:.3e} |"
            )

    lines.append("")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    lines.append(
        "## 5. Interpretation"
    )
    lines.append("")

    lines.append(
        "1. E9 confirms that a substantial fraction of the old observed "
        "method differences are unlikely to be explained by sample-level noise alone."
    )

    lines.append(
        "2. The old E9 result is exactly reproduced: "
        "**68/120 planned paired t-tests remain significant after "
        "within-family Bonferroni correction**."
    )

    lines.append(
        "3. For Qwen3.5-0.8B, SplitRAG's gains over both RAG and Decomp "
        "on 2i, 3i, and 2u remain highly significant after correction."
    )

    lines.append(
        "4. The result should not be interpreted as universal superiority: "
        "some comparisons are non-significant or significantly favor the earlier method, "
        "consistent with the model- and topology-dependent conclusions."
    )

    lines.append(
        "5. Wilcoxon is reported only as a robustness check; the primary "
        "120-test claim is based on paired t-tests."
    )

    lines.append("")

    return "\n".join(lines)
def summarize(base_dir,output_dir: Path,) -> Dict[str, Any]:

    output_dir.mkdir(parents=True,exist_ok=True,)
    sample_rows, formal_audit = collect_formal_records(base_dir)
    method_rows = summarize_methods(sample_rows)
    test_rows = paired_tests(sample_rows)
    significance_rows = significance_summary(test_rows)

    overall = next(
        r
        for r in significance_rows
        if r["scope_type"] == "overall"
    )


    write_json(
        output_dir
        / "data_audit.json",
        {
            "formal_audit":
                formal_audit,
        },
    )

    write_csv(
        output_dir
        / "method_summary.csv",
        method_rows,
        [
            "model",
            "qtype",
            "method",
            "n",
            "mean_f1",
            "mean_precision",
            "mean_recall",
            "exact_match_rate",
        ],
    )

    write_csv(
        output_dir
        / "paired_tests.csv",
        test_rows,
        [
            "model",
            "qtype",
            "comparison",
            "method_A",
            "method_B",
            "delta_direction",
            "n",
            "mean_f1_A",
            "mean_f1_B",
            "delta_f1_B_minus_A",
            "delta_95ci_low",
            "delta_95ci_high",
            "cohen_dz",
            "paired_t_stat",
            "paired_t_p_raw",
            "paired_t_p_bonferroni_family6",
            "paired_t_sig_bonf_0_05",
            "wilcoxon_stat",
            "wilcoxon_p_raw",
            "wilcoxon_p_bonferroni_family6",
            "wilcoxon_sig_bonf_0_05",
            "wins_B_gt_A",
            "ties",
            "losses_B_lt_A",
            "significant_direction",
        ],
    )

    write_csv(
        output_dir
        / "significance_summary.csv",
        significance_rows,
        [
            "scope_type",
            "scope",
            "n_tests",
            "paired_t_significant_after_family_bonf",
            "paired_t_significant_rate",
            "paired_t_significant_B_better",
            "paired_t_significant_B_worse",
            "wilcoxon_significant_after_family_bonf",
            "wilcoxon_significant_rate",
        ],
    )

    write_csv(
        output_dir
        / "sample_level_f1.csv",
        sample_rows,
        [
            "model",
            "method",
            "source_folder",
            "qtype",
            "sample_id",
            "exact_match",
            "precision",
            "recall",
            "f1",
            "pred_size",
            "gold_size",
        ],
    )

    summary = {

        "primary_test":
            "two-sided paired-sample t-test on per-query F1",

        "robustness_test":
            "two-sided Wilcoxon signed-rank test",

        "multiple_testing":
            {
                "family":
                    "within each model × qtype",

                "planned_comparisons_per_family":
                    6,

                "bonferroni_formula":
                    "min(p_raw * 6, 1)",

                "alpha":
                    0.05,

                "equivalent_raw_alpha_threshold":
                    0.05 / 6,
            },

        "planned_test_count":
            120,

        "primary_significant_after_bonferroni":
            overall[
                "paired_t_significant_after_family_bonf"
            ],

        "wilcoxon_significant_after_bonferroni":
            overall[
                "wilcoxon_significant_after_family_bonf"
            ],

        "formal_audit":
            formal_audit,


        "method_summary":
            method_rows,

        "paired_tests":
            test_rows,

        "significance_summary":
            significance_rows,
    }

    write_json(
        output_dir
        / "summary.json",
        summary,
    )

    report = build_markdown(
        formal_audit,
        method_rows,
        test_rows,
        significance_rows,
    )

    (
        output_dir
        / "summary.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    return summary
def main(base_dir,output_dir):
    summary = summarize(base_dir,output_dir)




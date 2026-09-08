from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, Sequence
DEFAULT_TYPES = ("1p", "2p", "2i", "2u", "3i")


class QueryError(ValueError):
    """Raised when a query does not match a supported symbolic form."""


@dataclass
class CheckResult:
    query_type: str
    case_id: str
    query: str
    predicted: list[int]
    expected: list[int]
    exact_match: bool
    missing: list[int]
    extra: list[int]
    error: str | None = None


class KnowledgeGraph:
    """Forward adjacency index keyed by (head entity, relation)."""

    def __init__(self) -> None:
        self.forward: DefaultDict[tuple[int, int], set[int]] = defaultdict(set)
        self.triple_count = 0

    @classmethod
    def from_csv(cls, path: Path) -> "KnowledgeGraph":
        graph = cls()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"head", "relation", "tail"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Graph CSV must contain columns {sorted(required)}; "
                    f"found {reader.fieldnames!r}"
                )
            for line_no, row in enumerate(reader, start=2):
                try:
                    head = int(row["head"])
                    relation = int(row["relation"])
                    tail = int(row["tail"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid triple at {path}:{line_no}: {row}") from exc
                graph.forward[(head, relation)].add(tail)
                graph.triple_count += 1
        return graph

    def project(self, entities: Iterable[int], relation: int) -> set[int]:
        result: set[int] = set()
        for entity in entities:
            result.update(self.forward.get((entity, relation), ()))
        return result


def parse_literal(text: str, source: Path) -> object:
    try:
        return ast.literal_eval(text.strip())
    except (SyntaxError, ValueError) as exc:
        raise QueryError(f"Cannot parse {source}: {exc}") from exc


def is_path_query(node: object) -> bool:
    return (
        isinstance(node, tuple)
        and len(node) == 2
        and isinstance(node[0], int)
        and isinstance(node[1], tuple)
        and len(node[1]) > 0
        and all(isinstance(relation, int) for relation in node[1])
    )


def execute(node: object, graph: KnowledgeGraph) -> set[int]:
    """Recursively evaluate a parsed benchmark query."""
    if is_path_query(node):
        anchor, relations = node
        entities = {anchor}
        for relation in relations:
            entities = graph.project(entities, relation)
            if not entities:
                break
        return entities

    if not isinstance(node, tuple) or not node:
        raise QueryError(f"Unsupported query node: {node!r}")

    # Union is encoded as (...subqueries..., ('u',)).
    if node[-1] == ("u",):
        operands = node[:-1]
        if len(operands) < 2:
            raise QueryError(f"Union requires at least two operands: {node!r}")
        result: set[int] = set()
        for operand in operands:
            result.update(execute(operand, graph))
        return result

    # A tuple of subqueries without an operator marker denotes intersection.
    if len(node) >= 2:
        evaluated = [execute(operand, graph) for operand in node]
        result = evaluated[0].copy()
        for values in evaluated[1:]:
            result.intersection_update(values)
        return result

    raise QueryError(f"Unsupported query shape: {node!r}")


def numeric_stem(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return sys.maxsize, path.stem


def check_type(benchmark: Path, query_type: str, graph: KnowledgeGraph) -> list[CheckResult]:
    query_dir = benchmark / query_type / "query_abstract"
    answer_dir = benchmark / query_type / "answer_abstract"
    if not query_dir.is_dir():
        raise FileNotFoundError(f"Missing query directory: {query_dir}")
    if not answer_dir.is_dir():
        raise FileNotFoundError(f"Missing answer directory: {answer_dir}")

    results: list[CheckResult] = []
    query_files = sorted(query_dir.glob("*.txt"), key=numeric_stem)
    for query_path in query_files:
        answer_path = answer_dir / query_path.name
        query_text = query_path.read_text(encoding="utf-8-sig").strip()
        predicted: set[int] = set()
        expected: set[int] = set()
        error: str | None = None
        try:
            query = parse_literal(query_text, query_path)
            answer = parse_literal(answer_path.read_text(encoding="utf-8-sig"), answer_path)
            if not isinstance(answer, set) or not all(isinstance(x, int) for x in answer):
                raise QueryError(f"Expected an integer set in {answer_path}, found {answer!r}")
            expected = answer
            predicted = execute(query, graph)
        except Exception as exc:  # Preserve per-case failures in the report.
            error = f"{type(exc).__name__}: {exc}"

        results.append(
            CheckResult(
                query_type=query_type,
                case_id=query_path.stem,
                query=query_text,
                predicted=sorted(predicted),
                expected=sorted(expected),
                exact_match=error is None and predicted == expected,
                missing=sorted(expected - predicted),
                extra=sorted(predicted - expected),
                error=error,
            )
        )
    return results


def write_reports(output_dir: Path, results: Sequence[CheckResult], graph: KnowledgeGraph) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(CheckResult.__dataclass_fields__)
    with (output_dir / "details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for field in ("predicted", "expected", "missing", "extra"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)

    with (output_dir / "mismatches.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            if not result.exact_match:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    by_type: dict[str, dict[str, int | float]] = {}
    for query_type in sorted({result.query_type for result in results}):
        subset = [result for result in results if result.query_type == query_type]
        matches = sum(result.exact_match for result in subset)
        errors = sum(result.error is not None for result in subset)
        by_type[query_type] = {
            "total": len(subset),
            "exact_matches": matches,
            "mismatches": len(subset) - matches,
            "errors": errors,
            "accuracy": matches / len(subset) if subset else 0.0,
        }

    total = len(results)
    matches = sum(result.exact_match for result in results)
    summary = {
        "graph_triples": graph.triple_count,
        "total": total,
        "exact_matches": matches,
        "mismatches": total - matches,
        "errors": sum(result.error is not None for result in results),
        "accuracy": matches / total if total else 0.0,
        "by_type": by_type,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark root")
    parser.add_argument("--graph", type=Path, required=True, help="CSV with head/relation/tail columns")
    parser.add_argument("--output", type=Path, default=Path("symbolic_check"), help="Report directory")
    parser.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES), help="Query types to check")
    return parser


def main() -> int:


    args = build_parser().parse_args()
    graph = KnowledgeGraph.from_csv(args.graph)
    results: list[CheckResult] = []
    for query_type in args.types:
        results.extend(check_type(args.benchmark, query_type, graph))
    summary = write_reports(args.output, results, graph)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

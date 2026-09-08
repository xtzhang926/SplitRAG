
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def read_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_kb_indices(kb: Dict[str, Any]):

    entities: Dict[str, Any] = kb["entities"]

    name_to_ids: Dict[str, List[str]] = defaultdict(list)

    relation_index: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    for entity_id, entity_data in entities.items():
        name = entity_data.get("name")
        if isinstance(name, str):
            name_to_ids[name].append(entity_id)

        for relation in entity_data.get("relations", []):
            predicate = relation.get("predicate")
            direction = relation.get("direction")
            object_id = relation.get("object")


            if not isinstance(predicate, str):
                continue
            if direction not in {"forward", "backward"}:
                continue
            if not isinstance(object_id, str):
                continue

            relation_index[(entity_id, predicate, direction)].add(object_id)

    return entities, dict(name_to_ids), relation_index


def reduce_program_to_relation_ast(program: Sequence[Dict[str, Any]], index: Optional[int] = None,) -> Tuple[Any, ...]:

    if index is None:
        index = len(program) - 1

    step = program[index]
    function = step.get("function")
    dependencies = step.get("dependencies", [])
    inputs = step.get("inputs", [])

    if function == "Find":
        if len(inputs) != 1:
            return ("X", "Find_invalid")
        return ("A", inputs[0])


    if function == "Relate":
        if len(dependencies) != 1 or len(inputs) < 2:
            return ("X", "Relate_invalid")

        predicate = inputs[0]
        direction = inputs[1]

        if direction not in {"forward", "backward"}:
            return ("X", "Relate_invalid_direction")

        child = reduce_program_to_relation_ast(program, dependencies[0])
        return ("P", predicate, direction, child)


    if function == "And":
        if len(dependencies) != 2:
            return ("X", "And_invalid")
        left = reduce_program_to_relation_ast(program, dependencies[0])
        right = reduce_program_to_relation_ast(program, dependencies[1])
        return ("I", left, right)

    if function == "Or":
        if len(dependencies) != 2:
            return ("X", "Or_invalid")
        left = reduce_program_to_relation_ast(program, dependencies[0])
        right = reduce_program_to_relation_ast(program, dependencies[1])
        return ("U", left, right)

    if function == "FilterConcept":
        if len(dependencies) != 1:
            return ("X", "FilterConcept_invalid")
        return reduce_program_to_relation_ast(program, dependencies[0])


    if function == "What":
        if len(dependencies) != 1:
            return ("X", "What_invalid")
        return reduce_program_to_relation_ast(program, dependencies[0])


    return ("X", str(function))
def ast_shape(ast_node: Tuple[Any, ...]) -> str:

    node_type = ast_node[0]

    if node_type == "A":
        return "A"
    if node_type == "P":
        return f"P({ast_shape(ast_node[3])})"
    if node_type == "I":
        return f"I({ast_shape(ast_node[1])},{ast_shape(ast_node[2])})"
    if node_type == "U":
        return f"U({ast_shape(ast_node[1])},{ast_shape(ast_node[2])})"

    return "X"
def flatten_binary_operator(ast_node: Tuple[Any, ...], operator: str) -> List[Tuple[Any, ...]]:

    if ast_node[0] == operator:
        return (
            flatten_binary_operator(ast_node[1], operator)
            + flatten_binary_operator(ast_node[2], operator)
        )
    return [ast_node]
def classify_splitrag_topology(ast_node: Tuple[Any, ...]) -> Optional[str]:

    shape = ast_shape(ast_node)

    if shape == "P(A)":
        return "1p"

    if shape == "P(P(A))":
        return "2p"

    if ast_node[0] == "I":
        branches = flatten_binary_operator(ast_node, "I")
        if all(ast_shape(branch) == "P(A)" for branch in branches):
            if len(branches) == 2:
                return "2i"
            if len(branches) == 3:
                return "3i"

    if ast_node[0] == "U":
        branches = flatten_binary_operator(ast_node, "U")
        if len(branches) == 2 and all(ast_shape(branch) == "P(A)" for branch in branches):
            return "2u"

    return None

def execute_relation_ast(ast_node: Tuple[Any, ...], name_to_ids: Dict[str, List[str]],
    relation_index: Dict[Tuple[str, str, str], Set[str]]) -> Set[str]:

    node_type = ast_node[0]

    if node_type == "A":
        entity_name = ast_node[1]
        return set(name_to_ids.get(entity_name, []))

    if node_type == "P":
        predicate = ast_node[1]
        direction = ast_node[2]
        child = ast_node[3]

        head_entities = execute_relation_ast(child, name_to_ids, relation_index)
        tails: Set[str] = set()
        for entity_id in head_entities:
            tails.update(relation_index.get((entity_id, predicate, direction), set()))
        return tails

    if node_type == "I":
        left = execute_relation_ast(ast_node[1], name_to_ids, relation_index)
        right = execute_relation_ast(ast_node[2], name_to_ids, relation_index)
        return left.intersection(right)

    if node_type == "U":
        left = execute_relation_ast(ast_node[1], name_to_ids, relation_index)
        right = execute_relation_ast(ast_node[2], name_to_ids, relation_index)
        return left.union(right)


    return set()
def entity_ids_to_names(entity_ids: Iterable[str], entities: Dict[str, Any]) -> Set[str]:

    names: Set[str] = set()
    for entity_id in entity_ids:
        entity = entities.get(entity_id)
        if entity is None:
            continue
        name = entity.get("name")
        if isinstance(name, str):
            names.add(name)
    return names

def relation_key(predicate: str, direction: str) -> str:

    return f"{predicate}::{direction}"
def branch_to_anchor_and_relations(ast_node: Tuple[Any, ...]):

    if ast_node[0] == "A":
        return ast_node[1], []

    if ast_node[0] == "P":
        anchor_name, relations = branch_to_anchor_and_relations(ast_node[3])
        relations = list(relations)
        relations.append(
            {
                "predicate": ast_node[1],
                "direction": ast_node[2],
                "relation_key": relation_key(ast_node[1], ast_node[2]),
            }
        )
        return anchor_name, relations

    raise ValueError(f"Node is not a path branch: {ast_node}")
def extract_query_components(ast_node: Tuple[Any, ...], qtype: str, name_to_ids: Dict[str, List[str]],) -> Dict[str, Any]:

    if qtype in {"1p", "2p"}:
        branch_nodes = [ast_node]
    elif qtype == "2i":
        branch_nodes = flatten_binary_operator(ast_node, "I")
    elif qtype == "3i":
        branch_nodes = flatten_binary_operator(ast_node, "I")
    elif qtype == "2u":
        branch_nodes = flatten_binary_operator(ast_node, "U")
    else:
        raise ValueError(f"Unsupported qtype: {qtype}")

    anchors = []
    branch_relations = []

    for branch in branch_nodes:
        anchor_name, relations = branch_to_anchor_and_relations(branch)
        anchor_ids = name_to_ids.get(anchor_name, [])

        anchors.append(
            {
                "name": anchor_name,
                "ids": list(anchor_ids),
                "unique_id": anchor_ids[0] if len(anchor_ids) == 1 else None,
            }
        )
        branch_relations.append(relations)

    if qtype == "1p":
        logical_query = [
            anchors[0]["unique_id"],
            [branch_relations[0][0]["relation_key"]],
        ]

    elif qtype == "2p":
        logical_query = [
            anchors[0]["unique_id"],
            [
                branch_relations[0][0]["relation_key"],
                branch_relations[0][1]["relation_key"],
            ],
        ]

    elif qtype in {"2i", "3i", "2u"}:
        logical_query = []
        for anchor, relations in zip(anchors, branch_relations):
            logical_query.append(
                [
                    anchor["unique_id"],
                    [relations[0]["relation_key"]],
                ]
            )


    else:
        logical_query = None

    return {
        "anchors": anchors,
        "relations": branch_relations,
        "logical_query": logical_query,
    }
def anchors_are_unique(ast_node: Tuple[Any, ...], name_to_ids: Dict[str, List[str]]) -> bool:

    if ast_node[0] == "A":
        return len(name_to_ids.get(ast_node[1], [])) == 1

    if ast_node[0] == "P":
        return anchors_are_unique(ast_node[3], name_to_ids)

    if ast_node[0] in {"I", "U"}:
        return (
            anchors_are_unique(ast_node[1], name_to_ids)
            and anchors_are_unique(ast_node[2], name_to_ids)
        )

    return False

def build_function_inventory(validation_data: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    function_counter: Counter = Counter()
    pattern_counter: Counter = Counter()
    or_successor_counter: Counter = Counter()

    for sample in validation_data:
        program = sample.get("program", [])
        functions = tuple(step.get("function") for step in program)

        function_counter.update(functions)
        pattern_counter[functions] += 1

        for source_index, step in enumerate(program):
            if step.get("function") != "Or":
                continue

            successors = []
            for target_step in program:
                if source_index in target_step.get("dependencies", []):
                    successors.append(target_step.get("function"))

            if successors:
                for successor in successors:
                    or_successor_counter[successor] += 1
            else:
                or_successor_counter["END"] += 1


    top_patterns = [
        {
            "count": count,
            "pattern": " -> ".join(str(x) for x in pattern),
        }
        for pattern, count in pattern_counter.most_common(100)
    ]

    return {
        "num_validation_samples": len(validation_data),
        "function_counts": dict(function_counter.most_common()),
        "or_successor_counts": dict(or_successor_counter.most_common()),
        "top_100_program_patterns": top_patterns,
    }

def collect_valid_candidates(validation_data: Sequence[Dict[str, Any]], entities: Dict[str, Any],
                             name_to_ids: Dict[str, List[str]], relation_index: Dict[Tuple[str, str, str], Set[str]], ):

    raw_topology_counts: Counter = Counter()
    unique_anchor_counts: Counter = Counter()
    gold_entity_counts: Counter = Counter()
    validated_counts: Counter = Counter()

    rejected_reasons: Counter = Counter()
    valid_candidates: List[Dict[str, Any]] = []


    ast_shape_counter: Counter = Counter()

    for val_index, sample in enumerate(validation_data):
        program = sample.get("program", [])

        if not program:
            rejected_reasons["empty_program"] += 1
            continue

        ast_node = reduce_program_to_relation_ast(program)
        shape = ast_shape(ast_node)
        ast_shape_counter[shape] += 1

        qtype = classify_splitrag_topology(ast_node)

        if qtype is None:
            rejected_reasons["not_target_topology"] += 1
            continue

        raw_topology_counts[qtype] += 1


        if not anchors_are_unique(ast_node, name_to_ids):
            rejected_reasons[f"{qtype}:ambiguous_anchor"] += 1
            continue

        unique_anchor_counts[qtype] += 1

        gold_answer = sample.get("answer")
        if not isinstance(gold_answer, str):
            rejected_reasons[f"{qtype}:gold_not_string"] += 1
            continue


        gold_answer_ids = name_to_ids.get(gold_answer, [])
        if len(gold_answer_ids) == 0:
            rejected_reasons[f"{qtype}:gold_not_entity"] += 1
            continue

        gold_entity_counts[qtype] += 1

        predicted_ids = execute_relation_ast(ast_node, name_to_ids, relation_index) # 求解传入的ast_node，返回id集合
        predicted_names = entity_ids_to_names(predicted_ids, entities)


        if predicted_names != {gold_answer}:
            rejected_reasons[f"{qtype}:reduced_query_not_equal_gold"] += 1
            continue

        validated_counts[qtype] += 1

        components = extract_query_components(ast_node, qtype, name_to_ids)


        validated_gold_ids = sorted(
            entity_id
            for entity_id in predicted_ids
            if entities.get(entity_id, {}).get("name") == gold_answer
        )
        if set(validated_gold_ids) != set(predicted_ids):
            continue

        record = {
            "e11_sample_id": None,
            "source_split": "KQAPro-val",
            "source_index": val_index,
            "qtype": qtype,
            "question": sample.get("question"),
            "gold_answer_name": gold_answer,
            "gold_answer_ids": validated_gold_ids,
            "anchors": components["anchors"],
            "relations": components["relations"],
            "logical_query": components["logical_query"],
            "reduced_ast_shape": shape,
            "reduced_answer_ids": sorted(predicted_ids),
            "reduced_answer_names": sorted(predicted_names),
            "gold_program": program,
            "sparql": sample.get("sparql"),
            "choices": sample.get("choices"),
        }

        valid_candidates.append(record)

    stats = {
        "raw_topology_counts": dict(raw_topology_counts),
        "unique_anchor_counts": dict(unique_anchor_counts),
        "gold_entity_counts": dict(gold_entity_counts),
        "validated_counts": dict(validated_counts),
        "rejected_reasons": dict(rejected_reasons.most_common()),
        "top_50_reduced_ast_shapes": [
            {"shape": shape, "count": count}
            for shape, count in ast_shape_counter.most_common(50)
        ],
    }

    return valid_candidates, stats
def sample_per_type(candidates: Sequence[Dict[str, Any]], target_per_type: int, seed: int):

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[record["qtype"]].append(record)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []

    for qtype in ["1p", "2p", "2i", "3i", "2u"]:
        pool = sorted(grouped.get(qtype, []), key=lambda x: x["source_index"])

        qtype_seed = seed + sum(ord(ch) for ch in qtype)
        rng = random.Random(qtype_seed)

        if len(pool) >= target_per_type:
            chosen = rng.sample(pool, target_per_type)
            chosen.sort(key=lambda x: x["source_index"])
        else:
            chosen = list(pool)
            warnings.append(
                f"{qtype}: only {len(pool)} strictly validated samples are available; "
                f"target_per_type={target_per_type}. No synthetic duplication was performed."
            )


        final_records = []
        for local_index, record in enumerate(chosen, start=1):
            copied = dict(record)
            copied["e11_sample_id"] = f"{qtype}_{local_index:04d}"
            final_records.append(copied)

        selected[qtype] = final_records

    return selected, warnings

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a strictly validated KQA Pro natural-language subset for "
        )
    )

    parser.add_argument(
        "--val",
        type=Path,
        required=True,
        help="Path to KQA Pro val.json",
    )
    parser.add_argument(
        "--kb",
        type=Path,
        required=True,
        help="Path to KQA Pro kb.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("e11_samples"),
        help="Output directory",
    )
    parser.add_argument(
        "--target-per-type",
        type=int,
        default=5000,
        help="Desired number of validated samples per query type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=926,
        help="Random seed for deterministic sampling",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    validation_data = read_json(args.val)
    kb = read_json(args.kb)

    entities, name_to_ids, relation_index = build_kb_indices(kb)


    inventory = build_function_inventory(validation_data)
    write_json(args.out_dir / "function_inventory.json", inventory)

    candidates, filter_stats = collect_valid_candidates(
        validation_data=validation_data,
        entities=entities,
        name_to_ids=name_to_ids,
        relation_index=relation_index,
    )


    write_jsonl(args.out_dir / "all_valid_candidates.jsonl", candidates)


    selected, warnings = sample_per_type(
        candidates=candidates,
        target_per_type=args.target_per_type,
        seed=args.seed,
    )

    selected_dir = args.out_dir / "selected"
    for qtype, records in selected.items():
        write_json(selected_dir / f"{qtype}.json", records)


    selected_counts = {qtype: len(records) for qtype, records in selected.items()}

    summary = {
        "input": {
            "val_path": str(args.val),
            "kb_path": str(args.kb),
            "target_per_type": args.target_per_type,
            "seed": args.seed,
        },
        "dataset": {
            "num_validation_samples": len(validation_data),
            "num_kb_entities": len(entities),
        },
        "filter_stats": filter_stats,
        "selected_counts": selected_counts,
        "warnings": warnings,
        "notes": [
            "Only Find/Relate/And/Or topology is modeled; FilterConcept and What are topology-transparent.",
            "Every retained sample is re-executed on kb.json after topology reduction.",
            "A sample is kept only when the reduced relation query returns exactly the KQA Pro gold answer name.",
            "Anchor names must map to exactly one KB entity ID.",
            "Relations preserve KQA Pro forward/backward direction using predicate::direction keys.",
            "No missing topology is synthesized or duplicated merely to reach target_per_type.",
        ],
    }

    write_json(args.out_dir / "summary.json", summary)


    for qtype in ["1p", "2p", "2i", "3i", "2u"]:
        count = filter_stats["validated_counts"].get(qtype, 0)
        selected_count = selected_counts.get(qtype, 0)
        print(f"      {qtype}: validated={count:4d}, selected={selected_count:4d}")


    print("\nKoPL Or successor counts:")
    for successor, count in inventory["or_successor_counts"].items():
        print(f"      Or -> {successor}: {count}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"      - {warning}")


if __name__ == "__main__":
    main()

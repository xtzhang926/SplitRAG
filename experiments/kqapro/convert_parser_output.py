import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



ALLOWED_QUERY_TYPES = {"1p", "2p", "2i"}
ALLOWED_DIRECTIONS = {"forward", "backward"}
def read_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def normalize_name(text: str) -> str:
   
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()
def build_entity_name_index(kb: Dict[str, Any]) -> Dict[str, List[str]]:


    entities = kb.get("entities")



    index = defaultdict(list)

    for entity_id, entity_info in entities.items():

        if not isinstance(entity_info, dict):
            continue

        name = entity_info.get("name")

        if not isinstance(name, str) or not name.strip():
            continue

        key = normalize_name(name)
        index[key].append(str(entity_id))


    return {
        key: sorted(set(ids))
        for key, ids in index.items()
    }
def build_relation_index(kb: Dict[str, Any]) -> Dict[str, List[str]]:


    relation_names = defaultdict(set)


    for section_name in ("entities", "concepts"):

        section = kb.get(section_name, {})

        if not isinstance(section, dict):
            continue

        for _, item_info in section.items():

            if not isinstance(item_info, dict):
                continue

            relations = item_info.get("relations", [])

            if not isinstance(relations, list):
                continue

            for relation_info in relations:

                if not isinstance(relation_info, dict):
                    continue

                predicate = relation_info.get("predicate")

                if not isinstance(predicate, str) or not predicate.strip():
                    continue

                relation_names[
                    normalize_name(predicate)
                ].add(predicate.strip())

    return {
        key: sorted(values)
        for key, values in relation_names.items()
    }
def parse_predicted_output(raw_output: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

    if isinstance(raw_output, dict):
        value = raw_output

    elif isinstance(raw_output, str):
        try:
            value = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return None, "invalid_json"

    else:
        return None, "invalid_output_type"

    if not isinstance(value, dict):
        return None, "top_level_not_object"

    if set(value.keys()) != {"type", "query"}:
        return None, "invalid_top_level_fields"

    parse_type = value["type"]
    parse_query = value["query"]

    if parse_type not in ALLOWED_QUERY_TYPES:
        return None, "invalid_qtype"

    if not isinstance(parse_query, list):
        return None, "query_not_list"

    expected_shape = {
        "1p": (1, [1]),
        "2p": (1, [2]),
        "2i": (2, [1, 1]),
    }

    expected_branch_num, expected_path_lengths = expected_shape[parse_type]

    if len(parse_query) != expected_branch_num:
        return None, "invalid_branch_count"


    for branch_index, branch in enumerate(parse_query):

        if not isinstance(branch, dict):
            return None, "branch_not_object"

        if set(branch.keys()) != {"anchor", "path"}:
            return None, "invalid_branch_fields"

        anchor = branch["anchor"]
        path = branch["path"]

        if not isinstance(anchor, str) or not anchor.strip():
            return None, "invalid_anchor"

        if not isinstance(path, list):
            return None, "path_not_list"

        if len(path) != expected_path_lengths[branch_index]:
            return None, "invalid_path_length"

        for step in path:

            if not isinstance(step, dict):
                return None, "path_step_not_object"

            if set(step.keys()) != {"relation", "direction"}:
                return None, "invalid_path_step_fields"

            relation = step["relation"]
            direction = step["direction"]

            if not isinstance(relation, str) or not relation.strip():
                return None, "invalid_relation"

            if direction not in ALLOWED_DIRECTIONS:
                return None, "invalid_direction"

    return value, None
def ground_entity(entity_name: str,entity_name_index: Dict[str, List[str]],) -> Tuple[Optional[str], Optional[str]]:
   

    key = normalize_name(entity_name)

    candidate_ids = entity_name_index.get(key, [])

    if len(candidate_ids) == 0:
        return None, "entity_not_found"

    if len(candidate_ids) > 1:
        return None, "entity_ambiguous"

    return candidate_ids[0], None
def ground_relation(relation_name: str,relation_index: Dict[str, List[str]],) -> Tuple[Optional[str], Optional[str]]:
 
    key = normalize_name(relation_name)

    candidates = relation_index.get(key, [])

    if len(candidates) == 0:
        return None, "relation_not_found"

    if len(candidates) > 1:
        return None, "relation_ambiguous"

    return candidates[0], None
def ground_branch(branch: Dict[str, Any],entity_name_index: Dict[str, List[str]],
                  relation_index: Dict[str, List[str]],) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:



    anchor_name = branch["anchor"]
    anchor_id, error = ground_entity(anchor_name,entity_name_index)

    if error is not None:
        return None, error


    grounded_path = []

    for step in branch["path"]:

        relation_name = step["relation"]
        direction = step["direction"]
        canonical_relation, error = ground_relation(relation_name,relation_index,)

        if error is not None:
            return None, error

 
        rel_key = f"{canonical_relation}::{direction}"

        grounded_path.append({
            "predicted_relation": relation_name,
            "canonical_relation": canonical_relation,
            "direction": direction,
            "rel_key": rel_key,
        })

    return {
        "anchor_name": anchor_name,
        "anchor_id": anchor_id,
        "path": grounded_path,
    }, None
def build_logical_query(parsed: Dict[str, Any],entity_name_index: Dict[str, List[str]],relation_index: Dict[str, List[str]],
                        ) -> Tuple[Optional[Any],Optional[List[Dict[str, Any]]],Optional[str],]:
   

    parse_type = parsed["type"]
    parse_query = parsed["query"]

    # ---------- 10.1 所有 branch 先 grounding ----------
    grounded_branches = []

    for branch in parse_query:

        grounded_branch, error = ground_branch(branch,entity_name_index,relation_index,)

        if error is not None:
            return None, None, error

        grounded_branches.append(grounded_branch)

    if parse_type == "1p":
        branch = grounded_branches[0]
        ent_id = branch["anchor_id"]
        rel_key = branch["path"][0]["rel_key"]
        logical_query = [ent_id,[rel_key],]
        return logical_query, grounded_branches, None

    if parse_type == "2p":
        branch = grounded_branches[0]
        ent_id = branch["anchor_id"]
        rel_key1 = branch["path"][0]["rel_key"]
        rel_key2 = branch["path"][1]["rel_key"]
        logical_query = [ent_id,[rel_key1,rel_key2,],]
        return logical_query, grounded_branches, None

    if parse_type == "2i":
        branch1 = grounded_branches[0]
        branch2 = grounded_branches[1]
        ent_id1 = branch1["anchor_id"]
        ent_id2 = branch2["anchor_id"]
        rel_key1 = branch1["path"][0]["rel_key"]
        rel_key2 = branch2["path"][0]["rel_key"]
        logical_query = [[ent_id1,[rel_key1],],[ent_id2,[rel_key2],],]
        return logical_query, grounded_branches, None

    return None, None, "unsupported_qtype"
def convert_record(record: Dict[str, Any],gold_qtype: str,entity_name_index,relation_index) -> Dict[str, Any]:
  

    sample_id = record.get("sample_id")
    source_index = record.get("source_index")
    question = record.get("queries")
    raw_output = record.get("answers")

    result = {
        "sample_id": sample_id,
        "source_index": source_index,
        "question": question,
        "gold_qtype": gold_qtype,


        "predicted_qtype": None,
        "raw_output": raw_output,
        "parsed_output": None,
        "grounded_query": None,
        "logical_query": None,
        "status": "failed",
        "failure_reason": None,
    }


    parsed, error = parse_predicted_output(raw_output)

    if error is not None:
        result["failure_reason"] = error
        return result

    result["parsed_output"] = parsed
    result["predicted_qtype"] = parsed["type"]


    logical_query, grounded_query, error = build_logical_query(parsed,entity_name_index,relation_index,)

    if error is not None:
        result["failure_reason"] = error
        return result

    result["grounded_query"] = grounded_query
    result["logical_query"] = logical_query
    result["status"] = "ok"

    return result
def convert_all(parser_answer_dir: Path, kb_path: Path, output_dir: Path,) -> None:
    


    kb = read_json(kb_path)

    entity_name_index = build_entity_name_index(kb)
    relation_index = build_relation_index(kb)


    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_by_gold_qtype = {}


    for gold_qtype in ("1p", "2p", "2i"):

        input_path = parser_answer_dir / f"{gold_qtype}.json"

        records = read_json(input_path)
        converted = []
        for record in records:
            result = convert_record(record,gold_qtype,entity_name_index,relation_index)
            converted.append(result)
            all_results.append(result)


        write_json(output_dir / f"{gold_qtype}.json",converted,)

        status_counter = Counter(item["status"]for item in converted)
        failure_counter = Counter(
            item["failure_reason"]
            for item in converted
            if item["status"] != "ok"
        )

        predicted_type_counter = Counter(
            item["predicted_qtype"]
            for item in converted
            if item["predicted_qtype"] is not None
        )

        summary_by_gold_qtype[gold_qtype] = {
            "total": len(converted),
            "grounding_ok": status_counter["ok"],
            "grounding_failed": status_counter["failed"],
            "grounding_success_rate": (
                status_counter["ok"] / len(converted)
                if converted
                else 0.0
            ),
            "predicted_qtype_distribution":
                dict(sorted(predicted_type_counter.items())),
            "failure_reasons":
                dict(sorted(failure_counter.items())),
        }


    write_json(output_dir / "all.json",all_results)
    failed_results = [
        item
        for item in all_results
        if item["status"] != "ok"
    ]

    write_json(output_dir / "failed.json",failed_results,)


    overall_status_counter = Counter(
        item["status"]
        for item in all_results
    )

    overall_failure_counter = Counter(
        item["failure_reason"]
        for item in all_results
        if item["status"] != "ok"
    )

    summary = {
        "input_parser_answer_dir":
            str(parser_answer_dir),

        "kb_path":
            str(kb_path),

        "total_samples":
            len(all_results),

        "entity_name_inventory_size":
            len(entity_name_index),

        "relation_inventory_size":
            len(relation_index),

        "grounding_ok":
            overall_status_counter["ok"],

        "grounding_failed":
            overall_status_counter["failed"],

        "grounding_success_rate": (
            overall_status_counter["ok"] / len(all_results)
            if all_results
            else 0.0
        ),

        "failure_reasons":
            dict(sorted(overall_failure_counter.items())),

        "by_gold_qtype":
            summary_by_gold_qtype,
    }

    write_json(output_dir / "summary.json",summary,)
if __name__ == "__main__":
    ans_dir = Path(r"")
    kb_path = Path(r"")
    output_dir = Path(r"")
    convert_all(ans_dir, kb_path, output_dir)

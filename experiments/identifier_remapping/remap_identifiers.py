import json
import os.path
import pickle
import re
from pathlib import Path


PROMPT_PATH = Path(r"")
KG_PATH = Path(r"")
OUTPUT_PATH = Path(r"")

QUERY_STRUCTS = ["1p", "2p", "2i", "3i", "2u"]
TASKS = ["Rag", "Decompose", "SplitRag"]

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


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
def load_id2ent_id2rel(data_path):
    with open(os.path.join(data_path, "id2ent.pkl"), "rb") as handle:
        id2ent = pickle.load(handle)
    with open(os.path.join(data_path, "id2rel.pkl"), "rb") as handle:
        id2rel = pickle.load(handle)
    return id2ent, id2rel
def split_prompt(text):

    if ANSWER_MARKER not in text:
        raise ValueError(f"Prompt ：{ANSWER_MARKER!r}")
    triplet_section, question_section = text.split(ANSWER_MARKER, maxsplit=1)
    return triplet_section, ANSWER_MARKER, question_section
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
def map_text(text, ent_map, rel_map):

    triplet_section, marker, question_section = split_prompt(text)


    def replace_triplet(match):
        head, relation, tail = map(int, match.groups())
        return f"({head + ent_map},{relation + rel_map},{tail + ent_map})"

    mapped_triplet_section = TRIPLET_PATTERN.sub(
        replace_triplet,
        triplet_section,
    )


    mapped_question_section = QUESTION_ENTITY_PATTERN.sub(
        lambda match: f"{match.group(1)}{int(match.group(2)) + ent_map}",
        question_section,
    )


    mapped_question_section = QUESTION_RELATION_PATTERN.sub(
        lambda match: f"{match.group(1)}{int(match.group(2)) + rel_map}",
        mapped_question_section,
    )

    return mapped_triplet_section + marker + mapped_question_section
def map_answer(answer, ent_map):
    answer_ids = [item.strip() for item in answer.split(",") if item.strip()]
    return ",".join(str(int(answer_id) + ent_map) for answer_id in answer_ids)
def map_prompt():
    id2ent, _ = load_id2ent_id2rel(KG_PATH)
    max_ent = max(id2ent.keys())

    ent_map = max_ent * 2 + 1
    rel_map = max_ent

    for task in TASKS:
        for query_type in QUERY_STRUCTS:
            input_dir = PROMPT_PATH / task / query_type
            for txt_path in input_dir.iterdir():
                if not txt_path.is_file() or txt_path.suffix.lower() != ".txt":
                    continue

                source = read_json(txt_path)
                queries = source["query"]
                answer = source["answer"]



                mapped_queries = [
                    map_text(text, ent_map, rel_map)
                    for text in queries
                ]

                result = {
                    "query": mapped_queries,

                    "answer": map_answer(answer, ent_map),
                }

                save_dir = OUTPUT_PATH / task / query_type
                save_dir.mkdir(parents=True, exist_ok=True)
                write_json(save_dir / txt_path.name, result)


if __name__ == "__main__":
    map_prompt()

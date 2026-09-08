from typing import Any, Dict, List, Tuple, Set, Optional
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
TARGET_QTYPES = ("1p", "2p", "2i")
def abstract_relation(kb,save_dir):
    rel_set = set()
    entities: Dict[str, Any] = kb["entities"]
    for entity_id, entity_data in entities.items():
        for relation in entity_data.get("relations", []):
            predicate = relation.get("predicate")
            direction = relation.get("direction")
            if not isinstance(predicate, str):
                continue
            if direction not in {"forward", "backward"}:
                continue
            rel_set.add(predicate)
    rel2id = dict()
    rel_num = len(rel_set)
    for idx,rel in enumerate(rel_set,start=1):
        rel2id[rel+'::forward'] = idx
        rel2id[rel+'::backward'] = idx+rel_num
    id2rel = {v:k for k,v in rel2id.items()}
    rel2id_path = save_dir/ 'rel2id.json'
    id2rel_path = save_dir/ 'id2rel.json'
    write_json(rel2id_path,rel2id)
    write_json(id2rel_path,id2rel)
def read_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: Path, obj: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def load_source_data(label,qtype,base_dir):
    if label not in ['train','val']:
        raise ValueError("label wrong")

    file_path = base_dir / f'{qtype}.json'
    data = read_json(file_path)
    return data
def build_kb_indices(kb: Dict[str, Any]):

    entities: Dict[str, Any] = kb["entities"]

    index: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    for entity_id, entity_data in entities.items():
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
            index[(entity_id, predicate, direction)].add(object_id)
    return index
def name_to_ids(kb):
    entities: Dict[str, Any] = kb["entities"]
    name2ids: Dict[str, List[str]] = defaultdict(list)
    for entity_id, entity_data in entities.items():
        name = entity_data.get("name")
        if isinstance(name, str):
            name2ids[name].append(entity_id)
    return name2ids
def get_triple_from_kb(index,ent_rel,qtype,rel2id):
    triples = set()
    split_dict = dict()
    if qtype == '1p':
        ent_id,rel_key = ent_rel
        rel,direction = rel_key.split("::")
        tail_ids = index.get((str(ent_id),str(rel),str(direction)),set())
        for tail_id in tail_ids:
            triples.add((str(ent_id),str(rel_key),str(tail_id)))
        split_dict[1] = triples
    elif qtype == '2p':
        ent_id, rel_key1, rel_key2 = ent_rel
        rel1, direction1 = rel_key1.split("::")
        rel2, direction2 = rel_key2.split("::")
        tail_ids = index.get((str(ent_id), str(rel1), str(direction1)), set())
        for tail_id in tail_ids:
            tails = index.get((str(tail_id), str(rel2), str(direction2)), set())
            if len(tails) != 0:
                triples.add((ent_id, rel_key1, tail_id))
                for tail in tails:
                    triples.add((tail_id, rel_key2, tail))
        split_dict[1] = set()
        split_dict[2] = set()
        for triple in triples:
            if ent_id == triple[0] or rel_key1 == triple[1]:
                split_dict[1].add(triple)
            if rel_key2 == triple[1]:
                split_dict[2].add(triple)
    elif qtype == '2i':
        ent_id1, ent_id2, rel_key1, rel_key2 = ent_rel
        rel1, direction1 = rel_key1.split("::")
        rel2, direction2 = rel_key2.split("::")
        tail_ids1 = index.get((str(ent_id1), str(rel1), str(direction1)), set())
        tail_ids2 = index.get((str(ent_id2), str(rel2), str(direction2)), set())
        tails = tail_ids1.intersection(tail_ids2)
        if len(tails) != 0:
            for tail in tails:
                triples.add((ent_id1, rel_key1, tail))
                triples.add((ent_id2, rel_key2, tail))
        split_dict[1] = set()
        split_dict[2] = set()
        split_dict[3] = set()
        for triple in triples:
            if ent_id1 == triple[0] or rel_key1 == triple[1]:
                split_dict[1].add(triple)
            if ent_id2 == triple[0] or rel_key2 == triple[1]:
                split_dict[2].add(triple)
    map_triples = set()
    map_split_dict = dict()
    for triple in triples:
        map_triples.add((triple[0].strip("Q"),rel2id[triple[1]],triple[2].strip("Q")))

    for phase,triples in split_dict.items():
        map_split_dict[phase] = set()
        for triple in triples:
            map_split_dict[phase].add((triple[0].strip("Q"), rel2id[triple[1]], triple[2].strip("Q")))
    return map_triples,map_split_dict
def parse_logical_query(logical_query, query_type):
    e1 = r1 = e2 = r2 = None
    if query_type == "1p":
        [e1,[r1,]] = logical_query
        return [e1,r1]
    if query_type == "2p":
        [e1,[r1,r2,]] = logical_query
        return [e1,r1,r2]
    if query_type == "2i":
        [[e1, [r1, ]],[e2, [r2,]]] = logical_query
        return [e1,e2,r1,r2]
def format_triples(triples,step_flag):
    premise_end_tag = "\n"
    if step_flag:
        premise_tag = ("If any (h, r, t) triplets are provided below, they "
                       "indicate that entity h is related to entity t by relation r. "
                       "Otherwise, ignore this section.\n")
        split_num = len(triples.keys())
        premise_list = []
        for i in range(1, split_num + 1):
            filtered_set = triples[i]
            texts = []
            for triple in filtered_set:
                texts.append(str(triple).strip().replace(" ", "").replace("'",""))
            premise = premise_tag + ",".join(texts) + "\n" + premise_end_tag
            premise_list.append(premise)
        return premise_list
    else:
        premise_tag = ("Given the following (h,r,t) triplets where"
                       " entity h is related to entity t by relation r;\n")
        texts = []
        for triple in triples:
            texts.append(str(triple).strip().replace(" ", "").replace("'",""))
        premise = premise_tag + ",".join(texts) + "\n" + premise_end_tag
        return [premise]
def map_query(ent_rel,qtype,rel2id):
    if qtype == '1p':
        [e1, r1] = ent_rel
        return [e1.strip('Q'),[rel2id[r1],]]
    elif qtype == '2p':
        [e1,r1,r2] = ent_rel
        return [e1.strip('Q'),[rel2id[r1],rel2id[r2],]]
    elif qtype == '2i':
        [e1,e2,r1,r2] = ent_rel
        return [[e1.strip('Q'), [rel2id[r1], ]],[e2.strip('Q'), [rel2id[r2],]]]
    else:
        raise ValueError("error")
class PromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
    def __generate_question_1p(self, logical_query):
        e1, r1 = parse_logical_query(logical_query, query_type="1p")
        entity = e1
        relation = r1
        return f"Which entities are connected to {entity} by relation {relation}?"
    def __generate_question_2p(self, logical_query):
        e1, r1, r2, = parse_logical_query(logical_query, query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2

        return f"""Let us assume that the set of entities E is connected to entity {entity} by relation \
{relation1}. Then, what are the entities connected to E by relation {relation2}?"""
    def __generate_question_2i(self, logical_query):
        e1, e2, r1, r2,= parse_logical_query(logical_query, query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. The set of entities F \
is connected to entity {entity2} by relation {relation2}. Then, which entities exist in both E and F?"""
    def __generate_question(self, logical_query, query_type):
        if query_type == "1p": return self.__generate_question_1p(logical_query)
        if query_type == "2p": return self.__generate_question_2p(logical_query)
        if query_type == "2i": return self.__generate_question_2i(logical_query)
    def generate_prompt(self, logical_query, query_type):
        question = (self.__generate_question(logical_query, query_type))
        return ["".join([self.question_tag, question, self.explain_tag])]
class StepPromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
    def __generate_question_1p(self,logical_query):
        e1, r1, = parse_logical_query(logical_query,query_type="1p")
        entity = e1
        relation = r1
        return [f"Which entities are connected to {entity} by relation {relation}?"]
    def __generate_question_2p(self,logical_query):
        e1, r1, r2, = parse_logical_query(logical_query,query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2
        return [f"Which entities are connected to {entity} by relation {relation1}?",
                f"Which entities are connected to entities in [PP1] by relation {relation2}?"]
    def __generate_question_2i(self,logical_query):
        e1, e2, r1, r2, = parse_logical_query(logical_query,query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return [f"Which entities are connected to {entity1} by relation {relation1}?",
                f"Which entities are connected to {entity2} by relation {relation2}?",
                f"Which entities exist in both sets [PP1] and [PP2]?"]
    def __generate_question(self, logical_query, query_type):
        if query_type=="1p": return self.__generate_question_1p(logical_query)
        if query_type=="2p": return self.__generate_question_2p(logical_query)
        if query_type=="2i": return self.__generate_question_2i(logical_query)
    def generate_prompt(self, logical_query, query_type):
        questions = self.__generate_question(logical_query, query_type)
        new_question = []
        for question in questions:
            new_question.append(self.question_tag + question + self.explain_tag)
        return new_question
def get_query(kb,save_dir,rel2id_path):

    index = build_kb_indices(kb)
    rel2id = read_json(rel2id_path)
    prompt_generator = PromptGenerator()
    step_prompt_generator = StepPromptGenerator()
    for qtype in TARGET_QTYPES:
        datas = load_source_data('val',qtype)
        for idx,data in enumerate(datas,start=1):
            query = data['logical_query']
            ent_rel = parse_logical_query(query,qtype)
            map_triples,map_step_triples = get_triple_from_kb(index,ent_rel,qtype,rel2id)
            premise = format_triples(map_triples,False)
            step_premise = format_triples(map_step_triples,True)
            ids_query = map_query(ent_rel,qtype,rel2id)
            prompt = prompt_generator.generate_prompt(ids_query,qtype)
            step_prompt = step_prompt_generator.generate_prompt(ids_query,qtype)
            rag = [premise[0] + prompt[0]]
            decompose = [premise[0] + prompt for prompt in step_prompt]
            splitrag = [premise + prompt for premise,prompt in zip(step_premise,step_prompt)]
            write_json(
                save_dir / 'Rag' / qtype / f'{idx}.txt',
                {
                    'sample_id':data['e11_sample_id'],
                    'source_split':data['source_split'],
                    'source_index':data['source_index'],
                    'query':rag,
                    'answer':','.join(data['gold_answer_ids']),
                }
            )
            write_json(
                save_dir / 'Decompose' / qtype / f'{idx}.txt',
                {
                    'sample_id': data['e11_sample_id'],
                    'source_split': data['source_split'],
                    'source_index': data['source_index'],
                    'query': decompose,
                    'answer':','.join(data['gold_answer_ids']),
                }
            )
            write_json(
                save_dir / 'SplitRag' / qtype / f'{idx}.txt',
                {
                    'sample_id': data['e11_sample_id'],
                    'source_split': data['source_split'],
                    'source_index': data['source_index'],
                    'query': splitrag,
                    'answer':','.join(data['gold_answer_ids']),
                }
            )
def find_ans(sample_id,source_index,gold_qtype,base_dir):

    files = read_json(base_dir / f'{gold_qtype}.json')
    for file in files:
        if file['e11_sample_id'] == sample_id:
            assert file['source_index'] == source_index
            return ','.join(file['gold_answer_ids'])
    return None
def parse_query_to_prompt(kb):
    save_dir = Path(r"")
    base_dir = Path(r"")
    index = build_kb_indices(kb)
    rel2id = read_json(Path(r""))
    prompt_generator = PromptGenerator()
    step_prompt_generator = StepPromptGenerator()
    files = read_json(base_dir / 'all.json')
    qtype_dict = defaultdict(list)
    for data in files:
        if data['status'] != 'ok':
            continue
        qtype_dict[data['predicted_qtype']].append(data)
    for predicted_qtype,datas in qtype_dict.items():
        for idx,data in enumerate(datas,start=1):
            query = data['logical_query']
            ent_rel = parse_logical_query(query, predicted_qtype)
            map_triples, map_step_triples = get_triple_from_kb(index, ent_rel, predicted_qtype, rel2id)
            premise = format_triples(map_triples, False)
            step_premise = format_triples(map_step_triples, True)
            ids_query = map_query(ent_rel, predicted_qtype, rel2id)
            prompt = prompt_generator.generate_prompt(ids_query, predicted_qtype)
            step_prompt = step_prompt_generator.generate_prompt(ids_query, predicted_qtype)
            rag = [premise[0] + prompt[0]]
            decompose = [premise[0] + prompt for prompt in step_prompt]
            splitrag = [premise + prompt for premise, prompt in zip(step_premise, step_prompt)]
            write_json(
                save_dir / 'Rag' / predicted_qtype / f'{idx}.txt',
                {
                    'sample_id': data['sample_id'],
                    'source_index': data['source_index'],
                    'query': rag,
                    'answer': find_ans(data['sample_id'],data['source_index'],data['gold_qtype']),
                }
            )
            write_json(
                save_dir / 'Decompose' / predicted_qtype / f'{idx}.txt',
                {
                    'sample_id': data['sample_id'],
                    'source_index': data['source_index'],
                    'query': decompose,
                    'answer': find_ans(data['sample_id'],data['source_index'],data['gold_qtype']),
                }
            )
            write_json(
                save_dir / 'SplitRag' / predicted_qtype / f'{idx}.txt',
                {
                    'sample_id': data['sample_id'],
                    'source_index': data['source_index'],
                    'query': splitrag,
                    'answer': find_ans(data['sample_id'],data['source_index'],data['gold_qtype']),
                }
            )
if __name__ == "__main__":
    kb = read_json(Path(r""))
    parse_query_to_prompt(kb)






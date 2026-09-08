import hashlib

import random

import argparse
import ast
import json
import pandas as pd
import re
from collections import deque, defaultdict
from pathlib import Path

QUERY_STRUCTS = ["1p", "2p", "2i", "3i", "2u"]
class StepPromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
        self.query_structs = QUERY_STRUCTS

    def __generate_question_1p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query,query_type="1p")
        entity = e1
        relation = r1
        return [f"Which entities are connected to {entity} by relation {relation}?"]
    def __generate_question_2p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query,query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2
        return [f"Which entities are connected to {entity} by relation {relation1}?",
                f"Which entities are connected to entities in [PP1] by relation {relation2}?"]
    def __generate_question_2i(self,logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query,query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return [f"Which entities are connected to {entity1} by relation {relation1}?",
                f"Which entities are connected to {entity2} by relation {relation2}?",
                f"Which entities exist in both sets [PP1] and [PP2]?"]
    def __generate_question_3i(self,logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query,query_type="3i")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return [f"Which entities are connected to {entity1} by relation {relation1}?",
                f"Which entities are connected to {entity2} by relation {relation2}?",
                f"Which entities are connected to {entity3} by relation {relation3}?",
                f"Which entities exist in both sets [PP1], [PP2] and [PP3]?"]
    def __generate_question_2u(self,logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query,query_type="2u")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return [f"Which entities are connected to {entity1} by relation {relation1}?",
                f"Which entities are connected to {entity2} by relation {relation2}?",
                f"What entities exist in either sets [PP1] or [PP2]?"]
    def __generate_question(self, logical_query, query_type):
        if query_type=="1p": return self.__generate_question_1p(logical_query)
        if query_type=="2p": return self.__generate_question_2p(logical_query)
        if query_type=="2i": return self.__generate_question_2i(logical_query)
        if query_type=="3i": return self.__generate_question_3i(logical_query)
        if query_type=="2u": return self.__generate_question_2u(logical_query)
    def generate_prompt(self, logical_query, query_type):
        questions = self.__generate_question(logical_query, query_type)
        new_question = []
        for question in questions:
            new_question.append(self.question_tag + question + self.explain_tag)
        return new_question
class PromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
        self.query_structs = QUERY_STRUCTS
    def __generate_question_1p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query, query_type="1p")
        entity = e1
        relation = r1
        return f"Which entities are connected to {entity} by relation {relation}?"
    def __generate_question_2p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query, query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2

        return f"""Let us assume that the set of entities E is connected to entity {entity} by relation \
{relation1}. Then, what are the entities connected to E by relation {relation2}?"""
    def __generate_question_2i(self, logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query, query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. The set of entities F \
is connected to entity {entity2} by relation {relation2}. Then, which entities exist in both E and F?"""
    def __generate_question_3i(self, logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query, query_type="3i")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. The set of entities F \
is connected to entity {entity2} by relation {relation2}. The set of entities G is connected to entity {entity3} by relation {relation3}. \
Then, which entities exist in both E, F and G?"""
    def __generate_question_2u(self, logical_query):
        e1, r1, e2, r2, e3, r3 = parse_logical_query(logical_query, query_type="2u")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. F is the set of \
entities connected to entity {entity2} by relation {relation2}. Then, what entities exist in either E or F?"""
    def __generate_question(self, logical_query, query_type):
        if query_type == "1p": return self.__generate_question_1p(logical_query)
        if query_type == "2p": return self.__generate_question_2p(logical_query)
        if query_type == "2i": return self.__generate_question_2i(logical_query)
        if query_type == "3i": return self.__generate_question_3i(logical_query)
        if query_type == "2u": return self.__generate_question_2u(logical_query)
    def generate_prompt(self, logical_query, query_type):
        question = (self.__generate_question(logical_query, query_type))
        return ["".join([self.question_tag, question, self.explain_tag])]
class PremiseGenerator:
    def __init__(self, entity_triplets, relation_triplets):
        self.premise_tag = "Given the following (h,r,t) triplets where entity h is related to entity t by relation r;\n"
        self.premise_end_tag = "\n"
        self.entity_triplets = entity_triplets
        self.relation_triplets = relation_triplets
        self.query_structs = QUERY_STRUCTS

    def __get_premise(self, entity_set, relation_set):

        kg_triplets = defaultdict(set)
        for entity in entity_set:
            for triplet in self.entity_triplets[entity]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)

        for relation in relation_set:
            for triplet in self.relation_triplets[relation]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)

        return kg_triplets
    def __filter_premise_1p(self, kg_triplets, entities, relations):
        e, r = entities[0], relations[0]
        tails = kg_triplets.get((e, r), set([]))
        filtered_set = set([])
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e, r, tail))
        return filtered_set
    def __filter_premise_2p(self, kg_triplets, entities, relations):
        e = entities[0]
        r1, r2 = relations
        entity_set = kg_triplets.get((e, r1), set([]))
        filtered_set = set([])
        for entity in entity_set:
            tails = kg_triplets.get((entity, r2), set([]))
            if len(tails) != 0:
                filtered_set.add((e, r1, entity))
                for tail in tails:
                    filtered_set.add((entity, r2, tail))
        return filtered_set
    def __filter_premise_2i(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        tails = entity_set1.intersection(entity_set2)
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e1, r1, tail))
                filtered_set.add((e2, r2, tail))
        return filtered_set
    def __filter_premise_3i(self, kg_triplets, entities, relations):
        e1, e2, e3 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        entity_set3 = kg_triplets.get((e3, r3), set([]))
        tails = entity_set1.intersection(entity_set2).intersection(entity_set3)
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e1, r1, tail))
                filtered_set.add((e2, r2, tail))
                filtered_set.add((e3, r3, tail))
        return filtered_set
    def __filter_premise_2u(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        for es1 in entity_set1:
            filtered_set.add((e1, r1, es1))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        for es2 in entity_set2:
            filtered_set.add((e2, r2, es2))
        return filtered_set
    def filter_premise(self, kg_triplets, entities, relations, query_type):
        if query_type == "1p": return self.__filter_premise_1p(kg_triplets, entities, relations)
        if query_type == "2p": return self.__filter_premise_2p(kg_triplets, entities, relations)
        if query_type == "2i": return self.__filter_premise_2i(kg_triplets, entities, relations)
        if query_type == "3i": return self.__filter_premise_3i(kg_triplets, entities, relations)
        if query_type == "2u": return self.__filter_premise_2u(kg_triplets, entities, relations)
    def generate_premise(self, entity_set, relation_set, query_type):
        # 根据query中的实体与关系，从KG中筛选数据，再根据query类型进行二次筛选，最后对数据进行格式化，从元组转为字符串
        kg_triplets = self.__get_premise(entity_set, relation_set)
        filtered_set = self.filter_premise(kg_triplets, entity_set, relation_set, query_type)

        texts1 = []

        for triplet in filtered_set:
            texts1.append(str(triplet).strip().replace(" ", ""))
        premise = self.premise_tag + ",".join(texts1) + "\n" + self.premise_end_tag
        return [premise]
class StepPremiseGenerator:
    def __init__(self, entity_triplets, relation_triplets):
        self.premise_tag = "If any (h, r, t) triplets are provided below, they indicate that entity h is related to entity t by relation r. Otherwise, ignore this section.\n"
        self.premise_end_tag = "\n"
        self.entity_triplets = entity_triplets
        self.relation_triplets = relation_triplets
        self.query_structs = QUERY_STRUCTS
    def __get_premise(self, entity_set, relation_set):

        kg_triplets = defaultdict(set)
        for entity in entity_set:
            for triplet in self.entity_triplets[entity]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)

        for relation in relation_set:
            for triplet in self.relation_triplets[relation]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)

        return kg_triplets
    def __filter_premise_1p(self, kg_triplets, entities, relations):
        e, r = entities[0], relations[0]
        tails = kg_triplets.get((e, r), set([]))
        filtered_set = set([])
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e, r, tail))
        return filtered_set
    def __filter_premise_2p(self, kg_triplets, entities, relations):
        e = entities[0]
        r1, r2 = relations
        entity_set = kg_triplets.get((e, r1), set([]))
        filtered_set = set([])
        for entity in entity_set:
            tails = kg_triplets.get((entity, r2), set([]))
            if len(tails) != 0:
                filtered_set.add((e, r1, entity))
                for tail in tails:
                    filtered_set.add((entity, r2, tail))
        return filtered_set
    def __filter_premise_2i(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        tails = entity_set1.intersection(entity_set2)
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e1, r1, tail))
                filtered_set.add((e2, r2, tail))
        return filtered_set
    def __filter_premise_3i(self, kg_triplets, entities, relations):
        e1, e2, e3 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        entity_set3 = kg_triplets.get((e3, r3), set([]))
        tails = entity_set1.intersection(entity_set2).intersection(entity_set3)
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e1, r1, tail))
                filtered_set.add((e2, r2, tail))
                filtered_set.add((e3, r3, tail))
        return filtered_set
    def __filter_premise_2u(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        for es1 in entity_set1:
            filtered_set.add((e1, r1, es1))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        for es2 in entity_set2:
            filtered_set.add((e2, r2, es2))
        return filtered_set
    def filter_premise(self, kg_triplets, entities, relations, query_type):
        if query_type == "1p": return self.__filter_premise_1p(kg_triplets, entities, relations)
        if query_type == "2p": return self.__filter_premise_2p(kg_triplets, entities, relations)
        if query_type == "2i": return self.__filter_premise_2i(kg_triplets, entities, relations)
        if query_type == "3i": return self.__filter_premise_3i(kg_triplets, entities, relations)
        if query_type == "2u": return self.__filter_premise_2u(kg_triplets, entities, relations)
    def split_filtered_set(self,entity_set,relation_set,filtered_set,q_type):
        split_dict = defaultdict(set)
        if q_type == "1p":
            split_dict[1] = set()
            split_dict[1] = filtered_set
        elif q_type == "2p":
            split_dict[1] = set()
            split_dict[2] = set()
            e1 = entity_set[0]
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if r2 == triplet[1]:
                    split_dict[2].add(triplet)
        elif q_type == "2i":
            split_dict[1] = set()
            split_dict[2] = set()
            split_dict[3] = set()
            e1, e2 = entity_set
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
            split_dict[3] = set()
        elif q_type == "3i":
            split_dict[1] = set()
            split_dict[2] = set()
            split_dict[3] = set()
            split_dict[4] = set()
            e1, e2, e3 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if e3 == triplet[0] or r3 == triplet[1]:
                    split_dict[3].add(triplet)
            split_dict[4] = set()
        elif q_type == "2u":
            split_dict[1] = set()
            split_dict[2] = set()
            split_dict[3] = set()
            e1, e2 = entity_set
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
            split_dict[3] = set()
        split_num = len(split_dict.keys())
        premise_list = []
        for i in range(1,split_num+1):
            filtered_set = split_dict[i]
            texts = []

            for triplet in filtered_set:
                texts.append(str(triplet).strip().replace(" ", ""))
            premise = self.premise_tag + ",".join(texts) + "\n" + self.premise_end_tag
            premise_list.append(premise)
        return premise_list
    def generate_premise(self, entity_set, relation_set, query_type):
        # 根据query中的实体与关系，从KG中筛选数据，再根据query类型进行二次筛选，最后对数据进行格式化，从元组转为字符串
        kg_triplets = self.__get_premise(entity_set, relation_set)
        filtered_set = self.filter_premise(kg_triplets, entity_set, relation_set, query_type)
        return self.split_filtered_set(entity_set,relation_set,filtered_set,query_type)
def parse_logical_query(logical_query, query_type):
    e1 = r1 = e2 = r2 = e3 = r3 = None
    if query_type == "1p": (e1, (r1,)) = logical_query
    if query_type == "2p": (e1, (r1, r2)) = logical_query
    if query_type == "2u": ((e1, (r1,)), (e2, (r2,)),(u,)) = logical_query
    if query_type == "3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
    if query_type == "2i": ((e1, (r1,)), (e2, (r2,))) = logical_query

    return [e1, r1, e2, r2, e3, r3]
def convert_wrong_topology(logical_query, true_qtype, wrong_qtype):
    if true_qtype == "2i" and wrong_qtype == "2u":
        branch1, branch2 = logical_query
        # -1仅作为union结构标记，实际prompt生成不使用它
        return (branch1, branch2, (-1,))

    if true_qtype == "2u" and wrong_qtype == "2i":
        branch1, branch2, _ = logical_query
        return (branch1, branch2)

    raise ValueError(
        f"Unsupported topology confusion: "
        f"{true_qtype} -> {wrong_qtype}"
    )
def set_rng(seed,ratio,qtype):
    value = f"{seed}|wrong_ratio={ratio}|qtype={qtype}|"
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
def get_ent_tr__rel_tr(data_path):
    def read_csv_to_iter(file_path, shuffle=False):
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if shuffle:
            df = df.sample(frac=1, random_state=100).reset_index(drop=True)
        heads = df['head'].tolist()
        relations = df['relation'].tolist()
        tails = df['tail'].tolist()
        triples = zip(heads, relations, tails)
        return triples
    entity_triples, relation_triples = {}, {}
    triples = read_csv_to_iter(data_path)
    for i, triple in enumerate(triples):
        e1, rel, e2 = triple
        if e1 in entity_triples:
            entity_triples[e1].add(triple)
        else:
            entity_triples[e1] = set([triple])

        if e2 in entity_triples:
            entity_triples[e2].add(triple)
        else:
            entity_triples[e2] = set([triple])

        if rel in relation_triples:
            relation_triples[rel].add(triple)
        else:
            relation_triples[rel] = set([triple])
    return entity_triples, relation_triples
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
def load_benchmark(qtype,num,benchmark_path):
    file_path = benchmark_path / qtype / 'query_abstract' / f"{num}.txt"
    answer_path = benchmark_path / qtype / 'answer_abstract' / f"{num}.txt"
    query = ast.literal_eval(file_path.read_text())
    answer = answer_path.read_text().strip().strip("{").strip("}")
    return query,answer
def wrong_topology(true_qtype,wrong_qtype,kg_path,save_path,benchmark_path):
    entity_triples, relation_triples = get_ent_tr__rel_tr(kg_path)
    prompt_generator = PromptGenerator()
    step_prompt_generator = StepPromptGenerator()
    premise_generator = PremiseGenerator(entity_triples, relation_triples)
    step_premise_generator = StepPremiseGenerator(entity_triples, relation_triples)

    for num in range(1,151):
        benchmark,answer = load_benchmark(true_qtype,num,benchmark_path)
        wrong_benchmark = convert_wrong_topology(benchmark,true_qtype, wrong_qtype)
        e1, r1, e2, r2, e3, r3 = parse_logical_query(wrong_benchmark, wrong_qtype)
        entity_set = list(filter(lambda x: x != None, [e1, e2, e3]))
        relation_set = list(filter(lambda x: x != None, [r1, r2, r3]))
        prompt = prompt_generator.generate_prompt(wrong_benchmark,wrong_qtype)
        step_prompt = step_prompt_generator.generate_prompt(wrong_benchmark,wrong_qtype)
        premise = premise_generator.generate_premise(entity_set,relation_set,wrong_qtype)
        step_premise = step_premise_generator.generate_premise(entity_set,relation_set,wrong_qtype)
        save_dir = save_path
        save_path = save_dir / f"{true_qtype}_to_{wrong_qtype}" / 'Rag' / f'{num}.txt'
        save_path.parent.mkdir(parents=True,exist_ok=True)
        result = {
            'query':[premise[0]+prompt[0]],
            'answer':answer
        }
        write_json(save_path,result)

        save_path = save_dir / f"{true_qtype}_to_{wrong_qtype}" / 'Decompose' / f'{num}.txt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'query': [premise[0]+son_prompt for son_prompt in step_prompt],
            'answer': answer
        }
        write_json(save_path, result)

        save_path = save_dir / f"{true_qtype}_to_{wrong_qtype}" / 'SplitRag' / f'{num}.txt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'query': [son_premise + son_prompt for son_premise,son_prompt in zip(step_premise,step_prompt)],
            'answer': answer
        }
        write_json(save_path, result)

def rebuild_query_with_relations(logical_query, qtype, relations):
    e1, r1, e2, r2, e3, r3 = parse_logical_query(
        logical_query,
        qtype,
    )

    if qtype == "1p":
        return (e1, (relations[0],))

    if qtype == "2p":
        return (e1, (relations[0], relations[1]))

    if qtype == "2i":
        return (
            (e1, (relations[0],)),
            (e2, (relations[1],)),
        )

    if qtype == "3i":
        return (
            (e1, (relations[0],)),
            (e2, (relations[1],)),
            (e3, (relations[2],)),
        )

    if qtype == "2u":
        return (
            (e1, (relations[0],)),
            (e2, (relations[1],)),
            logical_query[2],     # 保留union marker
        )
def wrong_relation(ratio,kg_path,save_path,benchmark_path):
    entity_triples, relation_triples = get_ent_tr__rel_tr(kg_path)
    prompt_generator = PromptGenerator()
    step_prompt_generator = StepPromptGenerator()
    premise_generator = PremiseGenerator(entity_triples, relation_triples)
    step_premise_generator = StepPremiseGenerator(entity_triples, relation_triples)

    for qtype in QUERY_STRUCTS:
        rng = set_rng(926,ratio,qtype)
        for num in range(1,151):
            benchmark, answer = load_benchmark(qtype, num,benchmark_path)
            e1, r1, e2, r2, e3, r3 = parse_logical_query(benchmark, qtype)
            entity_set = list(filter(lambda x: x != None, [e1, e2, e3]))
            relation_set = list(filter(lambda x: x != None, [r1, r2, r3]))
            wrong_relation_set = []
            for relation in relation_set:
                if rng.random() < ratio:
                    while True:
                        wrong_rel = rng.choice(list(relation_triples.keys()))
                        if wrong_rel != relation:
                            break
                    wrong_relation_set.append(wrong_rel)
                else:
                    wrong_relation_set.append(relation)
            wrong_benchmark = rebuild_query_with_relations(benchmark,qtype,wrong_relation_set)
            prompt = prompt_generator.generate_prompt(wrong_benchmark, qtype)
            step_prompt = step_prompt_generator.generate_prompt(wrong_benchmark, qtype)
            premise = premise_generator.generate_premise(entity_set, wrong_relation_set, qtype)
            step_premise = step_premise_generator.generate_premise(entity_set, wrong_relation_set, qtype)
            save_dir = save_path

            save_path = save_dir / f"{ratio}" / 'Rag' / qtype / f'{num}.txt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                'query': [premise[0] + prompt[0]],
                'answer': answer
            }
            write_json(save_path, result)

            save_path = save_dir / f"{ratio}" / 'Decompose' / qtype / f'{num}.txt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                'query': [premise[0] + son_prompt for son_prompt in step_prompt],
                'answer': answer
            }
            write_json(save_path, result)

            save_path = save_dir / f"{ratio}" / 'SplitRag' / qtype / f'{num}.txt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                'query': [son_premise + son_prompt for son_premise, son_prompt in zip(step_premise, step_prompt)],
                'answer': answer
            }
            write_json(save_path, result)

if __name__ == '__main__':
    kg_path, save_path, benchmark_path = Path("")
    wrong_relation(0.05,kg_path, save_path, benchmark_path)
    wrong_topology('2i','2u',kg_path, save_path, benchmark_path)


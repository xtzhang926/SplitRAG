import pandas as pd
from pathlib import Path

import ast
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
QUERY_STRUCTS = ["1p", "2p", "2i", "3i", "2u"]
TRIPLE_PATTERN = re.compile(
    r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
)
QUESTION_MARKER = "\n\nAnswer the question:\n"
def format_triple(triple):
    head, relation, tail = triple
    return f"({head},{relation},{tail})"
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
def extract_triples(prompt):

    premise_section = prompt.split(QUESTION_MARKER, maxsplit=1)[0]
    return {
        tuple(map(int, match.groups()))
        for match in TRIPLE_PATTERN.finditer(premise_section)
    }
def replace_premise(prompt, string_premise):


    premise_and_header, question = prompt.split(QUESTION_MARKER, maxsplit=1)
    if "\n" not in premise_and_header:
        raise ValueError("Prompt is missing the premise header line")

    header, _old_premise = premise_and_header.split("\n", maxsplit=1)
    return f"{header}\n{string_premise}{QUESTION_MARKER}{question}"
def load_premise(prompt_dir,qtype,number):
    d_txt = prompt_dir / 'Decompose' / qtype / f'{number}.txt'
    txt = read_json(d_txt)
    premise_set = extract_triples(txt['query'][0])
    split_dict = dict()
    s_txt = prompt_dir / 'SplitRag' / qtype / f'{number}.txt'
    queries = read_json(s_txt)['query']
    for phase,query in enumerate(queries):
        split_dict[phase+1] = extract_triples(query)

    return premise_set,split_dict
def get_relation_set(qtype,number,benchmark_path):
    query = ast.literal_eval((benchmark_path / qtype / 'query_abstract' / f'{number}.txt').read_text())
    e1 = r1 = e2 = r2 = e3 = r3 = None
    if qtype == "1p": (e1, (r1,)) = query
    if qtype == "2p": (e1, (r1, r2)) = query
    if qtype == "2i": ((e1, (r1,)), (e2, (r2,))) = query
    if qtype == "3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = query
    if qtype == "2u": ((e1, (r1,)), (e2, (r2,)), (u,)) = query
    return list(filter(lambda x: x != None, [r1, r2, r3]))
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
class NoisyGenerator:
    def __init__(self, entity_triplets, relation_triplets, noise_ratio=1.0,min_noise_num=1,max_noise_num=None,seed=926,):
        self.noise_ratio = noise_ratio
        self.min_noise_num = min_noise_num
        self.max_noise_num = max_noise_num
        self.seed = seed
        self.relation_pool = dict()
        self.relation_triplets = relation_triplets
        self.entity_triplets = entity_triplets
        self.all_triplet = tuple(sorted({
            triplet
            for triplets in relation_triplets.values()
            for triplet in triplets
        }))
    def get_noise_num(self, phase_num):

        if self.noise_ratio == 0:
            return 0
        noise_num = max(
            self.min_noise_num,
            math.ceil(phase_num * self.noise_ratio),
        )
        if self.max_noise_num is not None:
            noise_num = min(noise_num, self.max_noise_num)
        return noise_num
    def noise_distribution(self, noise_num):

        quotient, remainder = divmod(noise_num, 3)
        return [quotient + int(index < remainder) for index in range(3)]
    def rng(self, phase,qtype):
        value = (
            f"{self.seed}|phase={phase}|qtype={qtype}|"
            f"ratio={self.noise_ratio}"
        )
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))
    def sample_noise(self, candidates, sample_num, rng, selected):
        candidates = sorted(set(candidates) - selected)
        if sample_num >= len(candidates):
            return set(candidates)
        return set(rng.sample(candidates, sample_num))
    def get_ents(self,split_dict):
        ents_dict = dict()
        for phase,triplets in split_dict.items():
            ents_dict[phase] = set()
            for triplet in triplets:
                ents_dict[phase].add(triplet[0])
        return ents_dict
    def get_relation_pool(self, rel):
        if rel not in self.relation_pool:
            self.relation_pool[rel] = tuple(sorted(
                self.relation_triplets.get(rel, set())
            ))
        return self.relation_pool[rel]
    def sample_pool(self,pool,sample_num,predicate,rng,selected,):
        if sample_num <= 0 or not pool:
            return set()
        result = set()
        max_attempts = max(1000, sample_num * 100)
        attempts = 0
        while (
            len(result) < sample_num
            and attempts < max_attempts
        ):
            triplet = pool[rng.randrange(len(pool))]
            attempts += 1
            if (
                triplet not in selected
                and triplet not in result
                and predicate(triplet)
            ):
                result.add(triplet)

        if len(result) < sample_num:
            for triplet in pool:
                if (
                    triplet not in selected
                    and triplet not in result
                    and predicate(triplet)
                ):
                    result.add(triplet)
                    if len(result) == sample_num:
                        break
        return result
    def noise_p(self, ents, rel, noise_num, rng)->set:
        if noise_num == 0 or rel is None:
            return set()

        noise_num_list = self.noise_distribution(noise_num)
        selected = set()
        same_ent_wrong_rel = {
            triplet
            for ent in ents
            for triplet in self.entity_triplets.get(ent, set())
            if triplet[0] == ent and triplet[1] != rel
        }
        selected.update(self.sample_noise(same_ent_wrong_rel,noise_num_list[0],rng,selected,))

        relation_pool = self.get_relation_pool(rel)
        selected.update(self.sample_pool(relation_pool,noise_num_list[1],lambda triplet: triplet[0] not in ents,rng,selected,))

        selected.update(self.sample_pool(
            self.all_triplet,
            noise_num_list[2],
            lambda triplet: (triplet[0] not in ents and triplet[1] != rel),
            rng,
            selected,
        ))

        remaining = noise_num - len(selected)
        selected.update(self.sample_pool(
            self.all_triplet,
            remaining,
            lambda triplet: not (triplet[0] in ents and triplet[1] == rel),
            rng,
            selected,
        ))
        return selected
    def generate_noise_premise(self,filtered_set, split_dict, relation_set, qtype):
        r1,r2,r3 = (relation_set + [None] * 3)[:3]
        noise_dict = dict()
        noise_split_dict = dict()
        ents_dict = self.get_ents(split_dict)

        rel_phase = {
            "1p": [r1,],
            "2p": [r1,r2,],
            "2i": [r1,r2,None],
            "3i": [r1,r2,r3,None],
            "2u": [r1,r2,None],
        }
        rel = rel_phase[qtype]

        for phase in ents_dict.keys():
            ents = ents_dict[phase]
            noise_num = self.get_noise_num(len(split_dict[phase]))
            rng = self.rng(phase,qtype)
            noise_dict[phase] = self.noise_p(ents, rel[phase-1], noise_num, rng)


        for phase, premise in split_dict.items():
            noise_split_dict[phase] = set(premise)
        for phase, noise in noise_dict.items():
            noise_split_dict[phase].update(noise)

        noise_filtered_set = set(filtered_set)
        for noise in noise_dict.values():
            noise_filtered_set.update(noise)
        return noise_filtered_set,noise_split_dict,noise_dict
def swap_prompt(base_prompt, qtype,num,noise_filtered_set,noise_split_dict):
    r_prompt = read_json(base_prompt / 'Rag' / qtype / f"{num}.txt")['query']
    d_prompt = read_json(base_prompt / 'Decompose' / qtype / f"{num}.txt")['query']
    s_prompt = read_json(base_prompt / 'SplitRag' / qtype / f"{num}.txt")['query']
    answer = read_json(base_prompt / 'Rag' / qtype / f"{num}.txt")['answer']
    string_premise = ','.join(format_triple(triple) for triple in sorted(noise_filtered_set))
    string_step_premise = dict()
    for phase,premise in noise_split_dict.items():
        string_step_premise[phase] = ','.join(format_triple(triple) for triple in sorted(premise))

    mapped_r_prompt = [
        replace_premise(prompt, string_premise)
        for prompt in r_prompt
    ]

    mapped_d_prompt = [
        replace_premise(prompt, string_premise)
        for prompt in d_prompt
    ]


    mapped_s_prompt = [
        replace_premise(prompt, string_step_premise[phase])
        for phase, prompt in enumerate(s_prompt, start=1)
    ]
    noise_rag = {
        'query':mapped_r_prompt,
        'answer':answer
    }
    noise_decompose = {
        'query':mapped_d_prompt,
        'answer':answer
    }
    noise_split_rag = {
        'query':mapped_s_prompt,
        'answer':answer
    }
    return noise_rag,noise_decompose,noise_split_rag

def get_noise_prompt(prompt_dir,kg_path,save_dir,noise_ratio,benchmark_path,base_prompt):
    entity_triples, relation_triples = get_ent_tr__rel_tr(kg_path)
    noiser = NoisyGenerator(entity_triples, relation_triples,noise_ratio)
    for qtype in QUERY_STRUCTS:
        for num in range(1,151):
            relation_set = get_relation_set(qtype,num,benchmark_path)
            premise_set,split_dict = load_premise(prompt_dir,qtype,num)
            noise_filtered_set,noise_split_dict,noise_dict = noiser.generate_noise_premise(premise_set,split_dict,relation_set,qtype)
            noise_rag,noise_decompose,noise_split_rag = swap_prompt(base_prompt,qtype,num,noise_filtered_set,noise_split_dict)

            Path(save_dir / f'{noise_ratio}' / qtype).mkdir(parents=True, exist_ok=True)
            Path(save_dir / 'Rag' / f'{noise_ratio}' / qtype).mkdir(parents=True,exist_ok=True)
            Path(save_dir / 'Decompose' / f'{noise_ratio}' / qtype).mkdir(parents=True,exist_ok=True)
            Path(save_dir / 'SplitRag' / f'{noise_ratio}' / qtype).mkdir(parents=True,exist_ok=True)


            write_json(save_dir / f'{noise_ratio}' / qtype / f'{num}.txt', {
                phase:[noise for noise in noise_set] for phase,noise_set in noise_dict.items()
            })
            write_json(save_dir / 'Rag' / f'{noise_ratio}' / qtype / f'{num}.txt',noise_rag)
            write_json(save_dir / 'Decompose' / f'{noise_ratio}' / qtype / f'{num}.txt',noise_decompose)
            write_json(save_dir / 'SplitRag' / f'{noise_ratio}' / qtype / f'{num}.txt',noise_split_rag)



if __name__ == '__main__':
    prompt_dir = Path(r'')
    kg_path = Path(r"")
    benchmark_path = Path(r"")
    base_prompt = Path(r"")
    save_dir = Path(r'')
    get_noise_prompt(prompt_dir,kg_path,save_dir,0.0,benchmark_path,base_prompt)
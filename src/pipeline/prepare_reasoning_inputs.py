import ast
import json
import os
import random
from collections import defaultdict
QUERY_STRUCTS = {
            "1p": ('e', ('r',)),
            "2p": ('e', ('r', 'r')),
            "3p": ('e', ('r', 'r', 'r')),
            "2i": (('e', ('r',)), ('e', ('r',))),
            "3i": (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
            "ip": ((('e', ('r',)), ('e', ('r',))), ('r',)),
            "pi": (('e', ('r', 'r')), ('e', ('r',))),
            "2u": (('e', ('r',)), ('e', ('r',)), ('u',)),
            "up": ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
            "2in": (('e', ('r',)), ('e', ('r', 'n'))),
            "3in": (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
            "inp": ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)),
            "pin": (('e', ('r', 'r')), ('e', ('r', 'n'))),
            "pni": (('e', ('r', 'r', 'n')), ('e', ('r',))),
        }
class LogicalPromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
        self.query_structs = QUERY_STRUCTS
        self.reverse_query_structs = {v: k for k, v in self.query_structs.items()}

    def parse_logical_query(self, logical_query, query_type):
        """
        传入一个query及其类型，根据类型解析出query中的实体与关系
        :param logical_query:
        :param query_type:
        :return:
        """
        assert (
                    query_type in self.query_structs), f"Only the following {list(self.query_structs.keys())} are supported."

        e1 = r1 = e2 = r2 = e3 = r3 = None
        if query_type == "1p": (e1, (r1,)) = logical_query
        if query_type == "2p": (e1, (r1, r2)) = logical_query
        if query_type == "3p": (e1, (r1, r2, r3)) = logical_query
        if query_type == "2i": ((e1, (r1,)), (e2, (r2,))) = logical_query
        if query_type == "3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
        if query_type == "2in": ((e1, (r1,)), (e2, (r2, n))) = logical_query
        if query_type == "3in": ((e1, (r1,)), (e2, (r2,)), (e3, (r3, n))) = logical_query
        if query_type == "inp": (((e1, (r1,)), (e2, (r2, n))), (r3,)) = logical_query
        if query_type == "pin": ((e1, (r1, r2)), (e2, (r3, n))) = logical_query
        if query_type == "pni": ((e1, (r1, r2, n)), (e2, (r3,))) = logical_query
        if query_type == "ip": (((e1, (r1,)), (e2, (r2,))), (r3,)) = logical_query
        if query_type == "pi": ((e1, (r1, r2)), (e2, (r3,))) = logical_query
        if query_type == "2u": ((e1, (r1,)), (e2, (r2,)), (u,)) = logical_query
        if query_type == "up": (((e1, (r1,)), (e2, (r2,)), (u,)), (r3,)) = logical_query
        return [e1, r1, e2, r2, e3, r3]

    def __generate_question_1p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="1p")
        entity = e1
        relation = r1
        return f"Which entities are connected to {entity} by relation {relation}?"

    def __generate_question_2p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2

        return f"""Let us assume that the set of entities E is connected to entity {entity} by relation \
{relation1}. Then, what are the entities connected to E by relation {relation2}?"""

    def __generate_question_3p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="3p")
        entity = e1
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity} by relation {relation1}. The set of entities F \
is connected to entities in E by relation {relation2}. Then, what are the entities connected to F by relation {relation3}?"""

    def __generate_question_2i(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. The set of entities F \
is connected to entity {entity2} by relation {relation2}. Then, which entities exist in both E and F?"""

    def __generate_question_3i(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="3i")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. The set of entities F \
is connected to entity {entity2} by relation {relation2}. The set of entities G is connected to entity {entity3} by relation {relation3}. \
Then, which entities exist in both E, F and G?"""

    def __generate_question_2in(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2in")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. And F is the set of \
entities connected to entity {entity2} by any relation other than relation {relation2}. Then, which entities exist in both E and F?"""

    def __generate_question_3in(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="3in")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. F is the set of \
entities connected to entity {entity2} by relation {relation2}. And G is the set of entities connected to entity {entity3} by any relation other \
than relation {relation3}. Then, which entities exist in both E, F and G?"""

    def __generate_question_inp(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="inp")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. And F is the set of entities \
connected to entity {entity2} by any relation other than relation {relation2}. Then, which entities are connected to the entities that exist in \
both E and F through the relationship {relation3}?"""

    def __generate_question_pin(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="pin")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. F is the set of entities \
connected to entities in E by relation {relation2}. And G is the set of entities connected to entity {entity2} by any relation other \
than relation {relation3}. Then, which entities exist in both F and G?"""

    def __generate_question_pni(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="pni")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}, F is the set of entities \
connected to entities in E by any relation other than {relation2}. And G is the set of entities connected to entity {entity2} by relation \
{relation3}. Then, which entities exist in both F and G?"""

    def __generate_question_ip(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="ip")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}, F is the set of entities \
connected to entity {entity2} by relation {relation2}. And G is the set of entities exist in both E and F. Then, what are the entities \
connected to entities in set G by relation {relation3}?"""

    def __generate_question_pi(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="pi")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. F is the set of entities \
connected to entities in E by relation {relation2}. And G is the set of entities connected to entity {entity2} by relation {relation3}. \
Then, which entities exist in both F and G?"""

    def __generate_question_2u(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2u")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. F is the set of \
entities connected to entity {entity2} by relation {relation2}. Then, what entities exist in either E or F?"""

    def __generate_question_up(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="up")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} by relation {relation1}. F is the set of entities \
connected to entity {entity2} by relation {relation2}. And G is the set of entities exist in either E or F. Then, what are the entities connected \
to entities in G by relation {relation3}?"""

    def __generate_question_nin(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="nin")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is not connected to \
                entity {entity1} by relation {relation1}, F is the set of entities not\
                connected to entity {entity2} by relation {relation2} and G is the \
                set of entities in the intersection of E and F. \
                Then, what are the entities which are not in the set G? \
                """

    def __generate_question_nipn(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="nipn")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is not connected to entity {entity1} \
                by relation {relation1} and \
                F is the set of entities not connected to entity {entity2} by relation {relation2}. \
                G is the set of entities in the intersection of E and F. \
                Then, what are the entities not connected to entities in G by relation {relation3}? \
                """

    def __generate_question(self, logical_query, query_type):
        """
        根据输入query的类型，按照固定模式将其解析为提示词
        :param logical_query:
        :param query_type:
        :return:prompt about query
        """
        assert (
                    query_type in self.query_structs), f"Only the following {list(self.query_structs.keys())} are supported."

        if query_type == "1p": return self.__generate_question_1p(logical_query)
        if query_type == "2p": return self.__generate_question_2p(logical_query)
        if query_type == "3p": return self.__generate_question_3p(logical_query)
        if query_type == "2i": return self.__generate_question_2i(logical_query)
        if query_type == "3i": return self.__generate_question_3i(logical_query)
        if query_type == "2in": return self.__generate_question_2in(logical_query)
        if query_type == "3in": return self.__generate_question_3in(logical_query)
        if query_type == "inp": return self.__generate_question_inp(logical_query)
        if query_type == "pin": return self.__generate_question_pin(logical_query)
        if query_type == "pni": return self.__generate_question_pni(logical_query)
        if query_type == "ip": return self.__generate_question_ip(logical_query)
        if query_type == "pi": return self.__generate_question_pi(logical_query)
        if query_type == "2u": return self.__generate_question_2u(logical_query)
        if query_type == "up": return self.__generate_question_up(logical_query)
        if query_type == "nin": return self.__generate_question_nin(logical_query)
        if query_type == "nipn": return self.__generate_question_nipn(logical_query)

    def generate_prompt(self, logical_query, query_type):
        question = (self.__generate_question(logical_query, query_type))
        return "".join([self.question_tag, question, self.explain_tag])
class StepLogicalPromptGenerator:
    def __init__(self):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
        self.query_structs = QUERY_STRUCTS
        self.reverse_query_structs = {v: k for k, v in self.query_structs.items()}

    def parse_logical_query(self, logical_query, query_type):
        assert (
                    query_type in self.query_structs), f"Only the following {list(self.query_structs.keys())} are supported."
        e1 = r1 = e2 = r2 = e3 = r3 = None
        if query_type == "1p": (e1, (r1,)) = logical_query
        if query_type == "2p": (e1, (r1, r2)) = logical_query
        if query_type == "3p": (e1, (r1, r2, r3)) = logical_query
        if query_type == "2i": ((e1, (r1,)), (e2, (r2,))) = logical_query
        if query_type == "3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
        if query_type == "2in": ((e1, (r1,)), (e2, (r2, n))) = logical_query
        if query_type == "3in": ((e1, (r1,)), (e2, (r2,)), (e3, (r3, n))) = logical_query
        if query_type == "inp": (((e1, (r1,)), (e2, (r2, n))), (r3,)) = logical_query
        if query_type == "pin": ((e1, (r1, r2)), (e2, (r3, n))) = logical_query
        if query_type == "pni": ((e1, (r1, r2, n)), (e2, (r3,))) = logical_query
        if query_type == "ip": (((e1, (r1,)), (e2, (r2,))), (r3,)) = logical_query
        if query_type == "pi": ((e1, (r1, r2)), (e2, (r3,))) = logical_query
        if query_type == "2u": ((e1, (r1,)), (e2, (r2,)), (u,)) = logical_query
        if query_type == "up": (((e1, (r1,)), (e2, (r2,)), (u,)), (r3,)) = logical_query

        return [e1, r1, e2, r2, e3, r3]

    def __generate_question_1p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="1p")
        entity = e1
        relation = r1
        return {"1p": [f"Which entities are connected to {entity} by relation {relation}?"]}

    def __generate_question_2p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2
        return {"2p": [f"Which entities are connected to {entity} by relation {relation1}?",
                       f"Which entities are connected to entities in [PP1] by relation {relation2}?"]
                }

    def __generate_question_3p(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="3p")
        entity = e1
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"3p": [f"Which entities are connected to {entity} by relation {relation1}?",
                       f"Which entities are connected to entities in [PP1] by relation {relation2}?",
                       f"Which entities are connected to entities in [PP2] by relation {relation3}?"]
                }

    def __generate_question_2i(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return {"2i": [f"Which entities are connected to {entity1} by relation {relation1}?",
                       f"Which entities are connected to {entity2} by relation {relation2}?",
                       f"Which entities exist in both sets [PP1] and [PP2]?"]
                }

    def __generate_question_3i(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="3i")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"3i": [f"Which entities are connected to {entity1} by relation {relation1}?",
                       f"Which entities are connected to {entity2} by relation {relation2}?",
                       f"Which entities are connected to {entity3} by relation {relation3}?",
                       f"Which entities exist in both sets [PP1], [PP2] and [PP3]?"]
                }

    def __generate_question_2in(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2in")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return {"2in": [f"Which entities are connected to {entity1} by relation {relation1}?",
                        f"Which entities are connected to {entity2} by any relation other than {relation2}?",
                        f"Which entities exist in both sets [PP1] and [PP2]?"]
                }

    def __generate_question_3in(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="3in")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"3in": [f"Which entities are connected to {entity1} by relation {relation1}?",
                        f"Which entities are connected to {entity2} by relation {relation2}?",
                        f"Which entities are connected to {entity3} by any relation other than {relation3}?",
                        f"Which entities exist in both sets [PP1], [PP2] and [PP3]?"]
                }

    def __generate_question_inp(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="inp")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"inp": [f"Which entities are connected to {entity1} by relation {relation1}?",
                        f"Which entities are connected to {entity2} by any relation other than {relation2}?",
                        f"Which entities exist in both sets [PP1] and [PP2]?",
                        f"What are the entities connected to any entity in [PP3] by relation {relation3}?"]
                }

    def __generate_question_pin(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="pin")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"pin": [f"Which entities are connected to {entity1} by relation {relation1}?",
                        f"Which entities are connected to entity set in [PP1] by relation {relation2}?",
                        f"Which entities are connected to {entity2} by any relation other than {relation3}?",
                        f"Which entities exist in both sets [PP2] and [PP3]?"]
                }

    def __generate_question_pni(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="pni")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"pni": [f"Which entities are connected to {entity1} by relation {relation1}?",
                        f"Which entities are connected to any entity in [PP1] by any relation other than {relation2}?",
                        f"Which entities are connected to {entity2} by relation {relation3}?",
                        f"Which entities exist in both sets [PP2] and [PP3]?"]
                }

    def __generate_question_ip(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="ip")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"ip": [f"Which entities are connected to {entity1} by relation {relation1}?",
                       f"Which entities are connected to {entity2} by relation {relation2}?",
                       f"Which entities exist in both sets [PP1] and [PP2]?",
                       f"What are the entities connected to any entity in [PP3] by relation {relation3}?"]
                }

    def __generate_question_pi(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="pi")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"pi": [f"Which entities are connected to {entity1} by relation {relation1}?",
                       f"Which entities are connected to [PP1] by relation {relation2}?",
                       f"Which entities are connected to {entity2} by relation {relation3}?",
                       f"Which entities exist in both sets [PP2] and [PP3]?"]
                }

    def __generate_question_2u(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="2u")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return {"2u": [f"Which entities are connected to {entity1} by relation {relation1}?",
                       f"Which entities are connected to {entity2} by relation {relation2}?",
                       f"What entities exist in either sets [PP1] or [PP2]?"]
                }

    def __generate_question_up(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="up")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"up": [f"Which entities are connected to {entity1} by relation {relation1}?",
                       f"Which entities are connected to {entity2} by relation {relation2}?",
                       f"What entities exist in either sets [PP1] or [PP2]?",
                       f"Which entities are connected to any entity in [PP3] by relation {relation3}?", ]
                }

    def __generate_question_nin(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="nin")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return {"nin": [f"Which entities are not connected to {entity1} by relation {relation1}?",
                        f"Which entities are not connected to {entity2} by relation {relation2}?",
                        f"What are the entities in the intersection of entity sets [PP1] and [PP2]?",
                        f"Which entities are not in the set of entities [PP3]?", ]
                }

    def __generate_question_nipn(self, logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query, query_type="nipn")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return {"nipn": [f"Which entities are not connected to {entity1} by relation {relation1}?",
                         f"Which entities are not connected to {entity2} by relation {relation2}?",
                         f"What are the entities in the intersection of entity sets [PP1] and [PP2]?",
                         f"Which entities are not connected to any entity in [PP3] by relation {relation3}?", ]
                }

    def __generate_question(self, logical_query, query_type) -> dict[str, list[str]]:
        assert (
                    query_type in self.query_structs), f"Only the following {list(self.query_structs.keys())} are supported."
        if query_type == "1p": return self.__generate_question_1p(logical_query)
        if query_type == "2p": return self.__generate_question_2p(logical_query)
        if query_type == "3p": return self.__generate_question_3p(logical_query)
        if query_type == "2i": return self.__generate_question_2i(logical_query)
        if query_type == "3i": return self.__generate_question_3i(logical_query)
        if query_type == "2in": return self.__generate_question_2in(logical_query)
        if query_type == "3in": return self.__generate_question_3in(logical_query)
        if query_type == "inp": return self.__generate_question_inp(logical_query)
        if query_type == "pin": return self.__generate_question_pin(logical_query)
        if query_type == "pni": return self.__generate_question_pni(logical_query)
        if query_type == "ip": return self.__generate_question_ip(logical_query)
        if query_type == "pi": return self.__generate_question_pi(logical_query)
        if query_type == "2u": return self.__generate_question_2u(logical_query)
        if query_type == "up": return self.__generate_question_up(logical_query)
        if query_type == "nin": return self.__generate_question_nin(logical_query)
        if query_type == "nipn": return self.__generate_question_nipn(logical_query)

    def generate_prompt(self, logical_query, query_type):
        question = self.__generate_question(logical_query, query_type)
        return {"question_tag": self.question_tag,
                "question": question,
                "explain_tag": self.explain_tag}
class PremiseGenerator:
    def __init__(self, entity_triplets, relation_triplets):
        self.premise_tag = "Given the following (h,r,t) triplets where entity h is related to entity t by relation r;\n"
        self.premise_end_tag = "\n"
        self.entity_triplets = entity_triplets
        self.relation_triplets = relation_triplets
        self.query_structs = QUERY_STRUCTS

    def __get_premise(self, entity_set, relation_set, query_type):

        """
        对于传入的实体集合和关系集合中的每一个元素，在数据库中查找与其相关的三元组并作为结果返回
        :param entity_set:
        :param relation_set:
        :return:
        """
        kg_triplets = defaultdict(set)
        for entity in entity_set:
            for triplet in self.entity_triplets[entity]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)

        if query_type == "2in":
            n_relation_set = [relation_set[1]]
            relation_set = [relation_set[0]]

        elif query_type == "3in":
            n_relation_set = [relation_set[2]]
            relation_set = [relation_set[0], relation_set[1]]

        elif query_type == "inp":
            n_relation_set = [relation_set[1]]
            relation_set = [relation_set[0], relation_set[2]]

        elif query_type == "pin":
            n_relation_set = [relation_set[2]]
            relation_set = [relation_set[0], relation_set[1]]

        elif query_type == "pni":
            n_relation_set = [relation_set[1]]
            relation_set = [relation_set[0], relation_set[2]]
        else:
            n_relation_set = []

        for relation in relation_set:
            for triplet in self.relation_triplets[relation]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)
        for n_relation in n_relation_set:
            for rel, triplets in self.relation_triplets.items():
                if rel == n_relation:
                    continue
                for triplet in triplets:
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

    def __filter_premise_3p(self, kg_triplets, entities, relations):
        e = entities[0]
        r1, r2, r3 = relations
        entity_set1 = kg_triplets.get((e, r1), set([]))
        filtered_set = set([])
        for entity1 in entity_set1:
            entity_set2 = kg_triplets.get((entity1, r2), set([]))
            for entity in entity_set2:
                tails = kg_triplets.get((entity, r3), set([]))
                if len(tails) != 0:
                    filtered_set.add((e, r1, entity1))
                    filtered_set.add((entity1, r2, entity))
                    for tail in tails:
                        filtered_set.add((entity, r3, tail))
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

    def __filter_premise_2in(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = set([])
        neg_tail_map = {}
        for key in kg_triplets:  # 遍历kg_triplets中的键值
            if (key[0] == e2) and (key[1] != r2):  # 找到键值中实体为e2,关系为!r2的键值
                for tail in kg_triplets[key]:  # 根据键值访问实值序列
                    if tail not in neg_tail_map:
                        neg_tail_map[tail] = set()
                    neg_tail_map[tail].add(key[1])

                entity_set2 = entity_set2.union(kg_triplets[key])  # e2,!r2对应的实体组
        tails = entity_set1.intersection(entity_set2)
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e1, r1, tail))
                for rel in neg_tail_map[tail]:
                    filtered_set.add((e2, rel, tail))
        return filtered_set

    def __filter_premise_3in(self, kg_triplets, entities, relations):
        e1, e2, e3 = entities
        r1, r2, r3 = relations
        filtered_set = set()

        entity_set1 = kg_triplets.get((e1, r1), set())
        entity_set2 = kg_triplets.get((e2, r2), set())
        entity_set3 = set([])

        neg_tail_map = {}

        for key in kg_triplets:
            if (key[0] == e3) and (key[1] != r3):
                for tail in kg_triplets[key]:
                    if tail not in neg_tail_map:
                        neg_tail_map[tail] = set()
                    neg_tail_map[tail].add(key[1])

                entity_set3 = entity_set3.union(kg_triplets[key])

        valid_tails = entity_set1.intersection(entity_set2).intersection(entity_set3)

        for tail in valid_tails:
            filtered_set.add((e1, r1, tail))
            filtered_set.add((e2, r2, tail))
            for rel in neg_tail_map[tail]:  # 遍历所有相关关系
                filtered_set.add((e3, rel, tail))

        return filtered_set

    def __filter_premise_inp(self, kg_triplets, entities, relations):

        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = set([])

        tail_relations_map = {}
        for key in kg_triplets:
            if (key[0] == e2) and (key[1] != r2):
                for tail in kg_triplets[key]:
                    if tail not in tail_relations_map:
                        tail_relations_map[tail] = set()
                    tail_relations_map[tail].add(key[1])

                entity_set2 = entity_set2.union(kg_triplets[key])
        entity_set3 = entity_set1.intersection(entity_set2)
        for entity in entity_set3:
            tails = kg_triplets.get((entity, r3), set([]))
            if len(tails) != 0:
                filtered_set.add((e1, r1, entity))
                for rel in tail_relations_map[entity]:  # 遍历所有相关关系
                    filtered_set.add((e2, rel, entity))
                for tail in tails:
                    filtered_set.add((entity, r3, tail))
        return filtered_set

    def __filter_premise_pin(self, kg_triplets, entities, relations):

        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = set([])
        entity_path_map = {}
        for entity in entity_set1:
            entity_set2 = entity_set2.union(kg_triplets[(entity, r2)])
            for triplet in kg_triplets[(entity, r2)]:
                if triplet not in entity_path_map:
                    entity_path_map[triplet] = set()
                entity_path_map[triplet].add(entity)

        entity_set3 = set([])
        neg_tail_map = {}
        for key in kg_triplets:
            if (key[0] == e2) and (key[1] != r3):
                for tail in kg_triplets[key]:
                    if tail not in neg_tail_map:
                        neg_tail_map[tail] = set()
                    neg_tail_map[tail].add(key[1])
                entity_set3 = entity_set3.union(kg_triplets[key])
        tails = entity_set2.intersection(entity_set3)
        if len(tails) != 0:
            for tail in tails:
                for ent in entity_path_map[tail]:
                    filtered_set.add((e1, r1, ent))
                    filtered_set.add((ent, r2, tail))
                for rel in neg_tail_map[tail]:
                    filtered_set.add((e2, rel, tail))
        return filtered_set

    def __filter_premise_pni(self, kg_triplets, entities, relations):

        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])

        entity_set1 = kg_triplets.get((e1, r1), set([]))

        entity_set3 = kg_triplets.get((e2, r3), set([]))

        for entity in entity_set1:
            entity_set2 = set([])
            neg_tail_map = {}
            for key in kg_triplets:
                if (key[0] == entity) and (key[1] != r2):
                    for tail in kg_triplets[key]:
                        if tail not in neg_tail_map:
                            neg_tail_map[tail] = set()
                        neg_tail_map[tail].add(key[1])
                    entity_set2 = entity_set2.union(kg_triplets[key])
            tails = entity_set2.intersection(entity_set3)

            if len(tails) != 0:
                for tail in tails:
                    filtered_set.add((e1, r1, entity))
                    # print(neg_tail_map.keys())
                    for rel in neg_tail_map[tail]:
                        filtered_set.add((entity, rel, tail))
                    filtered_set.add((e2, r3, tail))

        return filtered_set

    def __filter_premise_ip(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map = defaultdict(set)
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        entity_set3 = entity_set1.intersection(entity_set2)
        tails = set([])
        for entity in entity_set3:
            tails = tails.union(kg_triplets[(entity, r3)])
            for tail in kg_triplets[(entity, r3)]:
                entity_path_map[tail].add(entity)
        if len(tails) != 0:
            for tail in tails:
                for t in entity_path_map[tail]:
                    filtered_set.add((e1, r1, t))
                    filtered_set.add((e2, r2, t))
                    filtered_set.add((t, r3, tail))
        return filtered_set

    def __filter_premise_pi(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set2 = set([])
        entity_path_map = defaultdict(set)
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        for entity in entity_set1:
            entity_set2 = entity_set2.union(kg_triplets[(entity, r2)])
            for es2 in kg_triplets[(entity, r2)]:
                entity_path_map[es2].add(entity)
        entity_set3 = kg_triplets.get((e2, r3), set([]))
        tails = entity_set2.intersection(entity_set3)
        if len(tails) != 0:
            for tail in tails:
                for t in entity_path_map[tail]:
                    filtered_set.add((e1, r1, t))
                    filtered_set.add((t, r2, tail))
                filtered_set.add((e2, r3, tail))
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

    def __filter_premise_up(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map1 = {}
        entity_path_map2 = {}
        entity_path_map3 = defaultdict(set)

        entity_set1 = kg_triplets.get((e1, r1), set([]))
        for es1 in entity_set1:
            entity_path_map1[es1] = e1

        entity_set2 = kg_triplets.get((e2, r2), set([]))
        for es2 in entity_set2:
            entity_path_map2[es2] = e2

        entity_set3 = entity_set1.union(entity_set2)
        tails = set([])

        for entity in entity_set3:
            tails = tails.union(kg_triplets[(entity, r3)])
            for tail in kg_triplets[(entity, r3)]:
                entity_path_map3[tail].add(entity)

        if len(tails) != 0:
            for tail in tails:
                for t in entity_path_map3[tail]:
                    filtered_set.add((t, r3, tail))

                    if t in entity_path_map1:
                        filtered_set.add((e1, r1, t))

                    if t in entity_path_map2:
                        filtered_set.add((e2, r2, t))

        return filtered_set

    def __filter_premise_nin(self, kg_triplets, entities, relations):
        return self.__filter_premise_2u(kg_triplets, entities, relations)

    def __filter_premise_nipn(self, kg_triplets, entities, relations):
        return self.__filter_premise_up(kg_triplets, entities, relations)

    def filter_premise(self, kg_triplets, entities, relations, query_type):
        """
        根据query类型对初筛的KG进行二次筛选
        :param kg_triplets:
        :param entities:
        :param relations:
        :param query_type:
        :return:三元组集合
        """
        assert (
                    query_type in self.query_structs), f"Only the following {list(self.query_structs.keys())} are supported."
        if query_type == "1p": return self.__filter_premise_1p(kg_triplets, entities, relations)
        if query_type == "2p": return self.__filter_premise_2p(kg_triplets, entities, relations)
        if query_type == "3p": return self.__filter_premise_3p(kg_triplets, entities, relations)
        if query_type == "2i": return self.__filter_premise_2i(kg_triplets, entities, relations)
        if query_type == "3i": return self.__filter_premise_3i(kg_triplets, entities, relations)
        if query_type == "2in": return self.__filter_premise_2in(kg_triplets, entities, relations)
        if query_type == "3in": return self.__filter_premise_3in(kg_triplets, entities, relations)
        if query_type == "inp": return self.__filter_premise_inp(kg_triplets, entities, relations)
        if query_type == "pin": return self.__filter_premise_pin(kg_triplets, entities, relations)
        if query_type == "pni": return self.__filter_premise_pni(kg_triplets, entities, relations)
        if query_type == "ip": return self.__filter_premise_ip(kg_triplets, entities, relations)
        if query_type == "pi": return self.__filter_premise_pi(kg_triplets, entities, relations)
        if query_type == "2u": return self.__filter_premise_2u(kg_triplets, entities, relations)
        if query_type == "up": return self.__filter_premise_up(kg_triplets, entities, relations)
        if query_type == "nin": return self.__filter_premise_nin(kg_triplets, entities, relations)
        if query_type == "nipn": return self.__filter_premise_nipn(kg_triplets, entities, relations)

    def generate_premise(self, entity_set, relation_set, query_type):
        # 根据query中的实体与关系，从KG中筛选数据，再根据query类型进行二次筛选，最后对数据进行格式化，从元组转为字符串
        kg_triplets = self.__get_premise(entity_set, relation_set, query_type)
        filtered_set = self.filter_premise(kg_triplets, entity_set, relation_set, query_type)

        texts1 = []

        for triplet in filtered_set:
            texts1.append(str(triplet).strip().replace(" ", ""))
        premise = self.premise_tag + ",".join(texts1) + "\n" + self.premise_end_tag
        return premise
class StepPremiseGenerator:
    def __init__(self, entity_triplets, relation_triplets):
        self.premise_tag = "If any (h, r, t) triplets are provided below, they indicate that entity h is related to entity t by relation r. Otherwise, ignore this section.\n"
        self.premise_end_tag = "\n"
        self.entity_triplets = entity_triplets
        self.relation_triplets = relation_triplets
        self.query_structs = QUERY_STRUCTS

    def __get_premise(self, entity_set, relation_set, query_type):

        """
        对于传入的实体集合和关系集合中的每一个元素，在数据库中查找与其相关的三元组并作为结果返回
        :param entity_set:
        :param relation_set:
        :return:
        """
        kg_triplets = defaultdict(set)
        for entity in entity_set:
            for triplet in self.entity_triplets[entity]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)

        if query_type == "2in":
            n_relation_set = [relation_set[1]]
            relation_set = [relation_set[0]]

        elif query_type == "3in":
            n_relation_set = [relation_set[2]]
            relation_set = [relation_set[0], relation_set[1]]

        elif query_type == "inp":
            n_relation_set = [relation_set[1]]
            relation_set = [relation_set[0], relation_set[2]]

        elif query_type == "pin":
            n_relation_set = [relation_set[2]]
            relation_set = [relation_set[0], relation_set[1]]

        elif query_type == "pni":
            n_relation_set = [relation_set[1]]
            relation_set = [relation_set[0], relation_set[2]]
        else:
            n_relation_set = []

        for relation in relation_set:
            for triplet in self.relation_triplets[relation]:
                h, r, t = triplet
                kg_triplets[(h, r)].add(t)
        for n_relation in n_relation_set:
            for rel, triplets in self.relation_triplets.items():
                if rel == n_relation:
                    continue
                for triplet in triplets:
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

    def __filter_premise_3p(self, kg_triplets, entities, relations):
        e = entities[0]
        r1, r2, r3 = relations
        entity_set1 = kg_triplets.get((e, r1), set([]))
        filtered_set = set([])
        for entity1 in entity_set1:
            entity_set2 = kg_triplets.get((entity1, r2), set([]))
            for entity in entity_set2:
                tails = kg_triplets.get((entity, r3), set([]))
                if len(tails) != 0:
                    filtered_set.add((e, r1, entity1))
                    filtered_set.add((entity1, r2, entity))
                    for tail in tails:
                        filtered_set.add((entity, r3, tail))
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

    def __filter_premise_2in(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = set([])
        neg_tail_map = {}
        for key in kg_triplets:  # 遍历kg_triplets中的键值
            if (key[0] == e2) and (key[1] != r2):  # 找到键值中实体为e2,关系为!r2的键值
                for tail in kg_triplets[key]:  # 根据键值访问实值序列
                    if tail not in neg_tail_map:
                        neg_tail_map[tail] = set()
                    neg_tail_map[tail].add(key[1])

                entity_set2 = entity_set2.union(kg_triplets[key])  # e2,!r2对应的实体组
        tails = entity_set1.intersection(entity_set2)
        if len(tails) != 0:
            for tail in tails:
                filtered_set.add((e1, r1, tail))
                for rel in neg_tail_map[tail]:
                    filtered_set.add((e2, rel, tail))
        return filtered_set

    def __filter_premise_3in(self, kg_triplets, entities, relations):
        e1, e2, e3 = entities
        r1, r2, r3 = relations
        filtered_set = set()

        entity_set1 = kg_triplets.get((e1, r1), set())
        entity_set2 = kg_triplets.get((e2, r2), set())
        entity_set3 = set([])

        neg_tail_map = {}

        for key in kg_triplets:
            if (key[0] == e3) and (key[1] != r3):
                for tail in kg_triplets[key]:
                    if tail not in neg_tail_map:
                        neg_tail_map[tail] = set()
                    neg_tail_map[tail].add(key[1])

                entity_set3 = entity_set3.union(kg_triplets[key])

        valid_tails = entity_set1.intersection(entity_set2).intersection(entity_set3)

        for tail in valid_tails:
            filtered_set.add((e1, r1, tail))
            filtered_set.add((e2, r2, tail))
            for rel in neg_tail_map[tail]:  # 遍历所有相关关系
                filtered_set.add((e3, rel, tail))

        return filtered_set

    def __filter_premise_inp(self, kg_triplets, entities, relations):

        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = set([])

        tail_relations_map = {}
        for key in kg_triplets:
            if (key[0] == e2) and (key[1] != r2):
                for tail in kg_triplets[key]:
                    if tail not in tail_relations_map:
                        tail_relations_map[tail] = set()
                    tail_relations_map[tail].add(key[1])

                entity_set2 = entity_set2.union(kg_triplets[key])
        entity_set3 = entity_set1.intersection(entity_set2)
        for entity in entity_set3:
            tails = kg_triplets.get((entity, r3), set([]))
            if len(tails) != 0:
                filtered_set.add((e1, r1, entity))
                for rel in tail_relations_map[entity]:  # 遍历所有相关关系
                    filtered_set.add((e2, rel, entity))
                for tail in tails:
                    filtered_set.add((entity, r3, tail))
        return filtered_set

    def __filter_premise_pin(self, kg_triplets, entities, relations):

        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = set([])
        entity_path_map = {}
        for entity in entity_set1:
            entity_set2 = entity_set2.union(kg_triplets[(entity, r2)])
            for triplet in kg_triplets[(entity, r2)]:
                if triplet not in entity_path_map:
                    entity_path_map[triplet] = set()
                entity_path_map[triplet].add(entity)

        entity_set3 = set([])
        neg_tail_map = {}
        for key in kg_triplets:
            if (key[0] == e2) and (key[1] != r3):
                for tail in kg_triplets[key]:
                    if tail not in neg_tail_map:
                        neg_tail_map[tail] = set()
                    neg_tail_map[tail].add(key[1])
                entity_set3 = entity_set3.union(kg_triplets[key])
        tails = entity_set2.intersection(entity_set3)
        if len(tails) != 0:
            for tail in tails:
                for ent in entity_path_map[tail]:
                    filtered_set.add((e1, r1, ent))
                    filtered_set.add((ent, r2, tail))
                for rel in neg_tail_map[tail]:
                    filtered_set.add((e2, rel, tail))
        return filtered_set

    def __filter_premise_pni(self, kg_triplets, entities, relations):

        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])

        entity_set1 = kg_triplets.get((e1, r1), set([]))

        entity_set3 = kg_triplets.get((e2, r3), set([]))

        for entity in entity_set1:
            entity_set2 = set([])
            neg_tail_map = {}
            for key in kg_triplets:
                if (key[0] == entity) and (key[1] != r2):
                    for tail in kg_triplets[key]:
                        if tail not in neg_tail_map:
                            neg_tail_map[tail] = set()
                        neg_tail_map[tail].add(key[1])
                    entity_set2 = entity_set2.union(kg_triplets[key])
            tails = entity_set2.intersection(entity_set3)

            if len(tails) != 0:
                for tail in tails:
                    filtered_set.add((e1, r1, entity))
                    # print(neg_tail_map.keys())
                    for rel in neg_tail_map[tail]:
                        filtered_set.add((entity, rel, tail))
                    filtered_set.add((e2, r3, tail))

        return filtered_set

    def __filter_premise_ip(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map = defaultdict(set)
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        entity_set2 = kg_triplets.get((e2, r2), set([]))
        entity_set3 = entity_set1.intersection(entity_set2)
        tails = set([])
        for entity in entity_set3:
            tails = tails.union(kg_triplets[(entity, r3)])
            for tail in kg_triplets[(entity, r3)]:
                entity_path_map[tail].add(entity)
        if len(tails) != 0:
            for tail in tails:
                for t in entity_path_map[tail]:
                    filtered_set.add((e1, r1, t))
                    filtered_set.add((e2, r2, t))
                    filtered_set.add((t, r3, tail))
        return filtered_set

    def __filter_premise_pi(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set2 = set([])
        entity_path_map = defaultdict(set)
        entity_set1 = kg_triplets.get((e1, r1), set([]))
        for entity in entity_set1:
            entity_set2 = entity_set2.union(kg_triplets[(entity, r2)])
            for es2 in kg_triplets[(entity, r2)]:
                entity_path_map[es2].add(entity)
        entity_set3 = kg_triplets.get((e2, r3), set([]))
        tails = entity_set2.intersection(entity_set3)
        if len(tails) != 0:
            for tail in tails:
                for t in entity_path_map[tail]:
                    filtered_set.add((e1, r1, t))
                    filtered_set.add((t, r2, tail))
                filtered_set.add((e2, r3, tail))
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

    def __filter_premise_up(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map1 = {}
        entity_path_map2 = {}
        entity_path_map3 = defaultdict(set)

        entity_set1 = kg_triplets.get((e1, r1), set([]))
        for es1 in entity_set1:
            entity_path_map1[es1] = e1

        entity_set2 = kg_triplets.get((e2, r2), set([]))
        for es2 in entity_set2:
            entity_path_map2[es2] = e2

        entity_set3 = entity_set1.union(entity_set2)
        tails = set([])

        for entity in entity_set3:
            tails = tails.union(kg_triplets[(entity, r3)])
            for tail in kg_triplets[(entity, r3)]:
                entity_path_map3[tail].add(entity)

        if len(tails) != 0:
            for tail in tails:
                for t in entity_path_map3[tail]:
                    filtered_set.add((t, r3, tail))

                    if t in entity_path_map1:
                        filtered_set.add((e1, r1, t))

                    if t in entity_path_map2:
                        filtered_set.add((e2, r2, t))

        return filtered_set

    def __filter_premise_nin(self, kg_triplets, entities, relations):
        return self.__filter_premise_2u(kg_triplets, entities, relations)

    def __filter_premise_nipn(self, kg_triplets, entities, relations):
        return self.__filter_premise_up(kg_triplets, entities, relations)

    def filter_premise(self, kg_triplets, entities, relations, query_type):
        """
        根据query类型对初筛的KG进行二次筛选
        :param kg_triplets:
        :param entities:
        :param relations:
        :param query_type:
        :return:三元组集合
        """
        assert (
                    query_type in self.query_structs), f"Only the following {list(self.query_structs.keys())} are supported."
        if query_type == "1p": return self.__filter_premise_1p(kg_triplets, entities, relations)
        if query_type == "2p": return self.__filter_premise_2p(kg_triplets, entities, relations)
        if query_type == "3p": return self.__filter_premise_3p(kg_triplets, entities, relations)
        if query_type == "2i": return self.__filter_premise_2i(kg_triplets, entities, relations)
        if query_type == "3i": return self.__filter_premise_3i(kg_triplets, entities, relations)
        if query_type == "2in": return self.__filter_premise_2in(kg_triplets, entities, relations)
        if query_type == "3in": return self.__filter_premise_3in(kg_triplets, entities, relations)
        if query_type == "inp": return self.__filter_premise_inp(kg_triplets, entities, relations)
        if query_type == "pin": return self.__filter_premise_pin(kg_triplets, entities, relations)
        if query_type == "pni": return self.__filter_premise_pni(kg_triplets, entities, relations)
        if query_type == "ip": return self.__filter_premise_ip(kg_triplets, entities, relations)
        if query_type == "pi": return self.__filter_premise_pi(kg_triplets, entities, relations)
        if query_type == "2u": return self.__filter_premise_2u(kg_triplets, entities, relations)
        if query_type == "up": return self.__filter_premise_up(kg_triplets, entities, relations)
        if query_type == "nin": return self.__filter_premise_nin(kg_triplets, entities, relations)
        if query_type == "nipn": return self.__filter_premise_nipn(kg_triplets, entities, relations)

    def split_filtered_set(self, entity_set, relation_set, filtered_set, q_type):
        split_dict = defaultdict(set)
        if q_type == "1p":
            split_dict[1] = filtered_set
        elif q_type == "2p":
            e1 = entity_set[0]
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if r2 == triplet[1]:
                    split_dict[2].add(triplet)
        elif q_type == "3p":
            e1 = entity_set[0]
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if r3 == triplet[1]:
                    split_dict[3].add(triplet)
        elif q_type == "2i":
            e1, e2 = entity_set
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
            split_dict[3] = set()
        elif q_type == "3i":
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
        elif q_type == "2in":
            e1, e2 = entity_set
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 != triplet[1]:
                    split_dict[2].add(triplet)
            split_dict[3] = set()
        elif q_type == "3in":
            e1, e2, e3 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if e3 == triplet[0] or r3 != triplet[1]:
                    split_dict[3].add(triplet)
            split_dict[4] = set()
        elif q_type == "inp":
            e1, e2 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 != triplet[1]:
                    split_dict[2].add(triplet)
                if r3 == triplet[1]:
                    split_dict[4].add(triplet)
            split_dict[3] = set()
        elif q_type == "pin":
            e1, e2 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if e2 == triplet[0] or r3 != triplet[1]:
                    split_dict[3].add(triplet)
            split_dict[4] = set()
        elif q_type == "pni":
            e1, e2 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if r2 != triplet[1]:
                    split_dict[2].add(triplet)
                if e2 == triplet[0] or r3 == triplet[1]:
                    split_dict[3].add(triplet)
            split_dict[4] = set()
        elif q_type == "ip":
            e1, e2 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if r3 == triplet[1]:
                    split_dict[4].add(triplet)
            split_dict[3] = set()
        elif q_type == "pi":
            e1, e2 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if e2 == triplet[0] or r3 == triplet[1]:
                    split_dict[3].add(triplet)
            split_dict[4] = set()
        elif q_type == "2u":
            e1, e2 = entity_set
            r1, r2 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
            split_dict[3] = set()
        elif q_type == "up":
            e1, e2 = entity_set
            r1, r2, r3 = relation_set
            for triplet in filtered_set:
                if e1 == triplet[0] or r1 == triplet[1]:
                    split_dict[1].add(triplet)
                if e2 == triplet[0] or r2 == triplet[1]:
                    split_dict[2].add(triplet)
                if r3 == triplet[1]:
                    split_dict[4].add(triplet)
            split_dict[3] = set()
        split_num = len(split_dict.keys())
        premise_list = []
        for i in range(1, split_num + 1):
            filtered_set = split_dict[i]
            texts = []

            for triplet in filtered_set:
                texts.append(str(triplet).strip().replace(" ", ""))
            premise = self.premise_tag + ",".join(texts) + "\n" + self.premise_end_tag
            premise_list.append(premise)
        return premise_list

    def generate_premise(self, entity_set, relation_set, query_type):
        # 根据query中的实体与关系，从KG中筛选数据，再根据query类型进行二次筛选，最后对数据进行格式化，从元组转为字符串
        kg_triplets = self.__get_premise(entity_set, relation_set, query_type)
        filtered_set = self.filter_premise(kg_triplets, entity_set, relation_set, query_type)
        return self.split_filtered_set(entity_set, relation_set, filtered_set, query_type)
def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
def load_q_a(q_name,base):
    qa_list = []
    aa_list = []
    qc_list = []
    ac_list = []
    base_path = base / q_name
    qa_folder = os.path.join(base_path,"query_abstract")
    aa_folder = os.path.join(base_path,"answer_abstract")
    qc_folder = os.path.join(base_path,"query_common")
    ac_folder = os.path.join(base_path,"answer_common")
    name_list = get_all_filenames(qa_folder)
    qa_path_list = [os.path.join(qa_folder,name) for name in name_list]
    aa_path_list = [os.path.join(aa_folder,name) for name in name_list]
    qc_path_list = [os.path.join(qc_folder, name) for name in name_list]
    ac_path_list = [os.path.join(ac_folder, name) for name in name_list]
    for i in range(len(qa_path_list)):
        with open(qa_path_list[i],"r",encoding="utf-8") as f:
            qa = f.read()
        with open(aa_path_list[i],"r",encoding="utf-8") as f:
            aa = f.read()
        with open(qc_path_list[i],"r",encoding="utf-8") as f:
            qc = f.read()
        with open(ac_path_list[i],"r",encoding="utf-8") as f:
            ac = f.read()
        qa_list.append(qa)
        aa_list.append(aa)
        qc_list.append(qc)
        ac_list.append(ac)
    return qa_list,aa_list,qc_list,ac_list
def get_ent_tr__rel_tr(data_path):
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
def process_queries(prompt_processor, premise_generator, logical_query, qtype, answer):
    """
    传入问题和答案集合，返回该问题对应的提示词与答案文本
    """
    e1, r1, e2, r2, e3, r3 = prompt_processor.parse_logical_query(logical_query, qtype)
    entity_set = list(filter(lambda x: x != None, [e1, e2, e3]))
    relation_set = list(filter(lambda x: x != None, [r1, r2, r3]))
    premise = premise_generator.generate_premise(entity_set, relation_set, qtype)
    question = prompt_processor.generate_prompt(logical_query, qtype)
    premise_question = "".join([premise, question])
    answer_text = answer.replace("{", "").replace("}", "").replace(" ", "")
    return {
        "query": premise_question,
        "answer": answer_text
    }
def process_step_queries(prompt_processor, premise_generator, logical_query, qtype, answer):
    e1, r1, e2, r2, e3, r3 = prompt_processor.parse_logical_query(logical_query, qtype)
    entity_set = list(filter(lambda x: x != None, [e1, e2, e3]))
    relation_set = list(filter(lambda x: x != None, [r1, r2, r3]))

    premise = premise_generator.generate_premise(entity_set, relation_set, qtype)
    question = prompt_processor.generate_prompt(logical_query, qtype)

    premise_question = {"premise": premise,
                        "question": question
                        }
    answer_text = answer.replace("{", "").replace("}", "").replace(" ", "")
    q_a = {"query": premise_question,
           "answer": answer_text
           }
    return q_a
def main(base_rag,base_decomp,base_splitrag,kg_path,base):
    entity_triplets, relation_triplets = get_ent_tr__rel_tr(kg_path)
    prompt_processor = LogicalPromptGenerator()
    step_prompt_processor = StepLogicalPromptGenerator()
    premise_generator = PremiseGenerator(entity_triplets, relation_triplets)
    step_premise_generator = StepPremiseGenerator(entity_triplets, relation_triplets)
    for q_name, s in QUERY_STRUCTS.items():

        rag = os.path.join(base_rag, f"{q_name}")
        decomp = os.path.join(base_decomp, f"{q_name}")
        splitrag = os.path.join(base_splitrag, f"{q_name}")
        os.makedirs(abstract, exist_ok=True)
        os.makedirs(step, exist_ok=True)
        os.makedirs(step_rag, exist_ok=True)
        qa_l, aa_l, qc_l, ac_l = load_q_a(q_name,base)
        for idx, q in enumerate(qa_l):
            q = ast.literal_eval(q)
            rag_q_a = process_queries(prompt_processor, premise_generator, q, q_name, aa_l[idx])
            decomp_q_a = process_step_queries(step_prompt_processor, premise_generator, q, q_name, aa_l[idx])
            splitrag_q_a = process_step_queries(step_prompt_processor, step_premise_generator, q, q_name, aa_l[idx])
            with open(os.path.join(rag, f"{idx + 1}.txt"), "w", encoding="utf-8") as f:
                f.write(str(rag_q_a))
            with open(os.path.join(decomp, f"{idx + 1}.txt"), "w", encoding="utf-8") as f:
                f.write(str(decomp_q_a))
            with open(os.path.join(splitrag, f"{idx + 1}.txt"), "w", encoding="utf-8") as f:
                f.write(str(splitrag_q_a))



if __name__ == '__main__':
    main()









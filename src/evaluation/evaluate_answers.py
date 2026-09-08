from time import sleep
import json
import os.path
import ast
import pickle
from pathlib import Path
QUERY_STRUCTS = ['1p', '2p', '2i', '3i', '2u']
def evaluate_entity_sets(pred,gold,parse_failed=False,status_failed=False,):
    pred = set() if pred is None else set(pred)
    gold = set() if gold is None else set(gold)

    if parse_failed or status_failed:
        return {
            "exact_match": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "missing_entities": sorted(gold),
            "extra_entities": sorted(pred),
            "parse_failed": int(parse_failed),
            "status_failed": int(status_failed),
        }

    if not pred and not gold:
        precision = recall = f1 = 1.0
    else:
        true_positive = len(pred & gold)
        precision = true_positive / len(pred) if pred else 0.0
        recall = true_positive / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

    return {
        "exact_match": int(pred == gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing_entities": sorted(gold - pred),
        "extra_entities": sorted(pred - gold),
        "parse_failed": 0,
        "status_failed": 0,
    }


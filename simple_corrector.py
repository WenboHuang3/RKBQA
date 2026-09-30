# -*- coding: utf-8 -*-
"""
Build a decision function: prescreen(pred_lf, threshold, min_freq) -> bool.
Rules:
- Extract [...] groups from pred_lf; treat groups with at least two nonempty
  comma-separated items as relations, ignoring other entities and constants.
- Trim whitespace, collapse repeated spaces, lowercase, and deduplicate relations
  within each sample.
- Estimate each relation's score as correct/support from labeled predictions.
- The revised-data prescreen averages scores with support >= min_freq.
- The direct-data prescreen requires every relation to meet min_freq and uses
  the minimum score. Return True to skip correction when the threshold is met.

Statistics can be built from either source:
1) revised.jsonl: predicted_lf and gold_field; "Correct logic form" marks correctness.
2) direct.jsonl: pred_lf and judge; judge=True marks correctness.
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# ---------- Relation parsing and normalization ----------
BRACKET_EXTRACT = re.compile(r"\[([^\[\]]+)\]")   # Extract the contents of each [...] group.
COMMA_SPLIT     = re.compile(r"\s*,\s*")
CORRECT_STR     = re.compile(r"^\s*Correct\s+logic\s+form\s*$", re.I)

def _norm_item(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # Collapse repeated whitespace.
    return s.lower()

def extract_relations_from_lf(pred_lf: str) -> List[str]:
    """
    Return normalized relations, deduplicated within the sample,
    such as 'sports, sports team roster, team'. A [...] group with at least two
    nonempty comma-separated items is a relation, with or without the 'R' operator.
    """
    if not isinstance(pred_lf, str) or not pred_lf:
        return []
    rels = []
    for m in BRACKET_EXTRACT.finditer(pred_lf):
        inside = m.group(1)
        parts = [_norm_item(x) for x in COMMA_SPLIT.split(inside)]
        parts = [p for p in parts if p]           # Remove empty items.
        if len(parts) >= 2:                       # Treat this group as a relation.
            rels.append(", ".join(parts))         # Use a consistent representation.
    return sorted(set(rels))                      # Deduplicate within the sample.

# ---------- Build relation statistics from revised.jsonl ----------
def _build_rel_stats_from_revised(revised_jsonl_path: str | Path) -> Dict[str, Dict[str, float]]:
    """
    Return {relation_str: {'support': int, 'correct': int, 'score': float}}.
    revised.jsonl must contain predicted_lf and gold_field.
    """
    revised_jsonl_path = Path(revised_jsonl_path)
    support = defaultdict(int)
    correct = defaultdict(int)

    with revised_jsonl_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred = (obj.get("predicted_lf") or "").strip()
            gold = (obj.get("gold_field") or "").strip()
            is_correct = bool(CORRECT_STR.match(gold))
            rels = extract_relations_from_lf(pred)
            if not rels: 
                continue
            for r in rels:            # Relations are already deduplicated per sample.
                support[r] += 1
                if is_correct:
                    correct[r] += 1

    stats = {}
    for r, t in support.items():
        c = correct.get(r, 0)
        stats[r] = {"support": t, "correct": c, "score": c/t if t else 0.0}
    return stats

# ---------- Build relation statistics from direct.jsonl (optional) ----------
def _build_rel_stats_from_direct(direct_jsonl_path: str | Path) -> Dict[str, Dict[str, float]]:
    """
    Return {relation_str: {'support': int, 'correct': int, 'score': float}}.
    direct.jsonl must contain pred_lf and judge (True means correct).
    """
    direct_jsonl_path = Path(direct_jsonl_path)
    support = defaultdict(int)
    correct = defaultdict(int)

    with direct_jsonl_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred = (obj.get("pred_lf") or "").strip()
            is_correct = bool(obj.get("judge", False))
            rels = extract_relations_from_lf(pred)
            if not rels:
                continue
            for r in rels:
                support[r] += 1
                if is_correct:
                    correct[r] += 1

    stats = {}
    for r, t in support.items():
        c = correct.get(r, 0)
        stats[r] = {"support": t, "correct": c, "score": c/t if t else 0.0}
    return stats

# ---------- Factory functions that return a prescreen callable ----------
def make_prescreen_func_from_revised(revised_jsonl_path: str | Path):
    """
    Build a decision function from revised.jsonl:
    prescreen(pred_lf: str, threshold: float, min_freq: int) -> bool
    Return True when the threshold is met (skip correction), otherwise False.
    """
    rel_stats = _build_rel_stats_from_revised(revised_jsonl_path)

    def prescreen(pred_lf: str, threshold: float, min_freq: int) -> bool:
        rels = extract_relations_from_lf(pred_lf)
        # Only use relations with support >= min_freq.
        usable = [rel_stats[r]["score"] for r in rels
                  if (r in rel_stats) and (rel_stats[r]["support"] >= min_freq)]
        if not usable:
            return False   # No usable relations: conservatively request correction.
        avg = sum(usable)/len(usable)
        return avg >= threshold

    return prescreen

def make_prescreen_func_from_direct(direct_jsonl_path: str | Path):
    """
    Build a decision function from direct.jsonl:
    prescreen(pred_lf: str, threshold: float, min_freq: int) -> bool
    """
    rel_stats = _build_rel_stats_from_direct(direct_jsonl_path)

    def prescreen(pred_lf: str, threshold: float, min_freq: int) -> bool:
        rels = extract_relations_from_lf(pred_lf)
        if any((r not in rel_stats) or (rel_stats[r]["support"] < min_freq) for r in rels):
            return False
        scores = [rel_stats[r]["score"] for r in rels]
        avg = min(scores) if scores else 0.0
        return avg >= threshold

    return prescreen

# ------------------- Usage example -------------------
if __name__ == "__main__":
    # Choose either the revised-data or direct-data factory.
    # prescreen = make_prescreen_func_from_revised("gpt4o_cot_responses_allerrors_revised_second.jsonl")
    DIRECT_STATS_JSON = "/mnt/public/algm/wb/data/gpt4o_cot_responses_allerrors_direct.jsonl" 

    prescreen = make_prescreen_func_from_direct(DIRECT_STATS_JSON)

    # Process one prediction:
    pred_lf = "( AND ( JOIN [ people , person , gender ] [ Male ] ) ( JOIN ( R [ people , person , children ] ) [ Jamie Spears ] ) )"
    is_high_conf = prescreen(pred_lf, threshold=0.85, min_freq=5)   
    print(is_high_conf)  # True/False

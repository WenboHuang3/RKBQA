# -*- coding: utf-8 -*-
"""
功能：构造一个判定函数 prescreen(pred_lf, threshold, min_freq) -> bool
规则：
- 从 pred_lf 中抽取所有 [...]；逗号分隔项数 >= 2 的视为“关系”（实体/常量通常 0/1 项，忽略）
- 关系字符串做规范化（去空白、折叠空格、小写），并在“样本内去重”
- 预先用带标注的数据统计“每个关系”的正确率 score = correct/support
- 对于输入的 pred_lf：只取 support >= min_freq 的关系，取这些关系 score 的“均值”
- 若均值 >= threshold 就返回 True（高于阈值 → 不修错），否则 False

你可以用两种来源构建统计：
1) revised.jsonl（字段：predicted_lf, gold_field），以 gold_field == "Correct logic form" 为正确
2) direct.jsonl（字段：pred_lf, judge），以 judge=True 为正确
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# ---------- 关系解析与规范化 ----------
BRACKET_EXTRACT = re.compile(r"\[([^\[\]]+)\]")   # 提取每个 [...] 的内容
COMMA_SPLIT     = re.compile(r"\s*,\s*")
CORRECT_STR     = re.compile(r"^\s*Correct\s+logic\s+form\s*$", re.I)

def _norm_item(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # 折叠多空格
    return s.lower()

def extract_relations_from_lf(pred_lf: str) -> List[str]:
    """
    返回“样本内去重 & 规范化”的关系列表（字符串形式，如 'sports , sports team roster , team'）
    只要 [...] 里逗号分隔项数 >= 2 就认作关系；是否带 'R' 不影响是否为关系。
    """
    if not isinstance(pred_lf, str) or not pred_lf:
        return []
    rels = []
    for m in BRACKET_EXTRACT.finditer(pred_lf):
        inside = m.group(1)
        parts = [_norm_item(x) for x in COMMA_SPLIT.split(inside)]
        parts = [p for p in parts if p]           # 去空元素
        if len(parts) >= 2:                       # 关系
            rels.append(", ".join(parts))         # 统一表示
    return sorted(set(rels))                      # 样本内去重

# ---------- 从 revised.jsonl 构建关系统计 ----------
def _build_rel_stats_from_revised(revised_jsonl_path: str | Path) -> Dict[str, Dict[str, float]]:
    """
    返回 { relation_str: {'support':int, 'correct':int, 'score':float} }
    revised.jsonl 需包含: predicted_lf, gold_field
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
            for r in rels:            # 样本内已去重
                support[r] += 1
                if is_correct:
                    correct[r] += 1

    stats = {}
    for r, t in support.items():
        c = correct.get(r, 0)
        stats[r] = {"support": t, "correct": c, "score": c/t if t else 0.0}
    return stats

# ---------- 从 direct.jsonl 构建关系统计（可选） ----------
def _build_rel_stats_from_direct(direct_jsonl_path: str | Path) -> Dict[str, Dict[str, float]]:
    """
    返回 { relation_str: {'support':int, 'correct':int, 'score':float} }
    direct.jsonl 需包含: pred_lf, judge（True=正确）
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

# ---------- 工厂函数：返回你要的“一个函数” ----------
def make_prescreen_func_from_revised(revised_jsonl_path: str | Path):
    """
    基于 revised.jsonl 构建判定函数：
    prescreen(pred_lf: str, threshold: float, min_freq: int) -> bool
    返回 True 表示“高于阈值（不修错）”，False 否则。
    """
    rel_stats = _build_rel_stats_from_revised(revised_jsonl_path)

    def prescreen(pred_lf: str, threshold: float, min_freq: int) -> bool:
        rels = extract_relations_from_lf(pred_lf)
        # 只用 support >= min_freq 的关系
        usable = [rel_stats[r]["score"] for r in rels
                  if (r in rel_stats) and (rel_stats[r]["support"] >= min_freq)]
        if not usable:
            return False   # 保守：无可用关系 → 不达阈值
        avg = sum(usable)/len(usable)
        return avg >= threshold

    return prescreen

def make_prescreen_func_from_direct(direct_jsonl_path: str | Path):
    """
    基于 direct.jsonl 构建判定函数：
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

# ------------------- 用法示例 -------------------
if __name__ == "__main__":
    # 任选其一：从 revised 或 direct 构建
    # prescreen = make_prescreen_func_from_revised("gpt4o_cot_responses_allerrors_revised_second.jsonl")
    DIRECT_STATS_JSON = "/mnt/public/algm/wb/data/gpt4o_cot_responses_allerrors_direct.jsonl" 

    prescreen = make_prescreen_func_from_direct(DIRECT_STATS_JSON)

    # 处理单条：
    pred_lf = "( AND ( JOIN [ people , person , gender ] [ Male ] ) ( JOIN ( R [ people , person , children ] ) [ Jamie Spears ] ) )"
    is_high_conf = prescreen(pred_lf, threshold=0.85, min_freq=5)   
    print(is_high_conf)  # True/False
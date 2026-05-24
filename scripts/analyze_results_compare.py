#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比三套结果（DW vs donut_only vs whisper_only）的对/错分布，并导出样例。

输入假设（与你当前 results/ 下的文件一致）：
  - results/DW_small/DW_small.json
  - results/donut_only_zh/donut_only_zh.json
  - results/whisper_only_zh/whisper_only_zh.json

每个 json 都是一个 list[dict]，包含字段：
  - id: str
  - ref: str
  - pred: str
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


ZH_PUNC_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "：": ":",
        "；": ";",
        "、": ",",
        "《": "<",
        "》": ">",
    }
)


def normalize_text(
    s: str,
    *,
    lower: bool = False,
    remove_spaces: bool = True,
    normalize_punc: bool = True,
    remove_surrounding_quotes: bool = True,
) -> str:
    if s is None:
        s = ""
    s = str(s)
    if normalize_punc:
        s = s.translate(ZH_PUNC_MAP)
    if remove_spaces:
        # 包括全角空格
        s = re.sub(r"[\s\u3000]+", "", s)
    if lower:
        s = s.lower()
    if remove_surrounding_quotes:
        # 只去掉最外层的一对引号，避免把内部内容破坏
        if len(s) >= 2:
            pairs = [('"', '"'), ("'", "'"), ("<", ">"), ("「", "」"), ("『", "』")]
            for l, r in pairs:
                if s.startswith(l) and s.endswith(r):
                    s = s[1:-1]
                    break
    return s


def edit_distance(a: List[str], b: List[str]) -> int:
    # 标准 DP，O(len(a)*len(b))，样本句子都很短，足够用
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[m]


def align_ops(ref: List[str], hyp: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    基于 Levenshtein DP 回溯得到一条最短对齐路径。
    返回 op 序列，每个元素为 (op, ref_tok_or_None, hyp_tok_or_None)
      - op in {"match", "sub", "del", "ins"}
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Tuple[int, int, str]]] = [[(0, 0, "")] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = (0, j - 1, "ins")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            # del / ins / sub-or-match
            cand = [
                (dp[i - 1][j] + 1, (i - 1, j, "del")),
                (dp[i][j - 1] + 1, (i, j - 1, "ins")),
                (dp[i - 1][j - 1] + cost, (i - 1, j - 1, "match" if cost == 0 else "sub")),
            ]
            best_cost, best_bt = min(cand, key=lambda x: x[0])
            dp[i][j] = best_cost
            bt[i][j] = best_bt

    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, op = bt[i][j]
        if op == "match" or op == "sub":
            ops.append((op, ref[i - 1], hyp[j - 1]))
        elif op == "del":
            ops.append((op, ref[i - 1], None))
        elif op == "ins":
            ops.append((op, None, hyp[j - 1]))
        else:
            raise RuntimeError(f"未知 op={op}")
        i, j = pi, pj
    ops.reverse()
    return ops


def cer(ref: str, hyp: str) -> float:
    # 中文默认用字符级
    r = list(ref)
    h = list(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / float(len(r))


def wer(ref: str, hyp: str) -> float:
    # 词级：按空白切（如果 normalize 去空白，则不适合用这个）
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / float(len(r))


def read_result_json(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 list JSON，实际类型={type(data)}")
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {idx} 条不是 dict，实际={type(row)}")
        if "id" not in row:
            raise ValueError(f"{path} 第 {idx} 条缺少字段 id")
        sid = str(row["id"])
        out[sid] = row
    return out


@dataclass
class JoinedRow:
    sid: str
    ref: str
    pred_dw: str
    pred_donut: str
    pred_whisper: str
    ok_dw: bool  # 兼容旧字段：句子级 exact match（仅用于调试/回溯）
    ok_donut: bool
    ok_whisper: bool
    cer_dw: Optional[float] = None
    cer_donut: Optional[float] = None
    cer_whisper: Optional[float] = None
    # 字符级统计（按 ref 字符计）
    ref_len: int = 0
    match_dw: int = 0
    sub_dw: int = 0
    del_dw: int = 0
    ins_dw: int = 0
    match_donut: int = 0
    sub_donut: int = 0
    del_donut: int = 0
    ins_donut: int = 0
    match_whisper: int = 0
    sub_whisper: int = 0
    del_whisper: int = 0
    ins_whisper: int = 0


def join_by_id(
    dw: Dict[str, Dict[str, Any]],
    donut: Dict[str, Dict[str, Any]],
    whisper: Dict[str, Dict[str, Any]],
    *,
    do_normalize: bool,
    norm_kwargs: Dict[str, Any],
    compute_cer: bool,
) -> Tuple[List[JoinedRow], Dict[str, List[str]]]:
    ids = sorted(set(dw) | set(donut) | set(whisper))
    missing = {
        "missing_in_dw": [i for i in ids if i not in dw],
        "missing_in_donut": [i for i in ids if i not in donut],
        "missing_in_whisper": [i for i in ids if i not in whisper],
    }

    rows: List[JoinedRow] = []
    for sid in ids:
        if sid not in dw or sid not in donut or sid not in whisper:
            continue
        ref = str(dw[sid].get("ref", ""))  # ref 理论上三者一致，用 dw 为准
        pdw = str(dw[sid].get("pred", ""))
        pdo = str(donut[sid].get("pred", ""))
        pwh = str(whisper[sid].get("pred", ""))

        if do_normalize:
            ref_n = normalize_text(ref, **norm_kwargs)
            pdw_n = normalize_text(pdw, **norm_kwargs)
            pdo_n = normalize_text(pdo, **norm_kwargs)
            pwh_n = normalize_text(pwh, **norm_kwargs)
        else:
            ref_n, pdw_n, pdo_n, pwh_n = ref, pdw, pdo, pwh

        ok_dw = pdw_n == ref_n
        ok_donut = pdo_n == ref_n
        ok_whisper = pwh_n == ref_n

        # 字符级：对齐到 ref 的每个字符，match 计为正确，其余（sub/del）计为错误
        ref_chars = list(ref_n)
        dw_ops = align_ops(ref_chars, list(pdw_n))
        donut_ops = align_ops(ref_chars, list(pdo_n))
        whisper_ops = align_ops(ref_chars, list(pwh_n))

        def count_ops(ops: List[Tuple[str, Optional[str], Optional[str]]]) -> Tuple[int, int, int, int]:
            c = Counter(op for op, _, _ in ops)
            return int(c.get("match", 0)), int(c.get("sub", 0)), int(c.get("del", 0)), int(c.get("ins", 0))

        m_dw, s_dw, d_dw, i_dw = count_ops(dw_ops)
        m_do, s_do, d_do, i_do = count_ops(donut_ops)
        m_wh, s_wh, d_wh, i_wh = count_ops(whisper_ops)

        row = JoinedRow(
            sid=sid,
            ref=ref,
            pred_dw=pdw,
            pred_donut=pdo,
            pred_whisper=pwh,
            ok_dw=ok_dw,
            ok_donut=ok_donut,
            ok_whisper=ok_whisper,
            ref_len=len(ref_chars),
            match_dw=m_dw,
            sub_dw=s_dw,
            del_dw=d_dw,
            ins_dw=i_dw,
            match_donut=m_do,
            sub_donut=s_do,
            del_donut=d_do,
            ins_donut=i_do,
            match_whisper=m_wh,
            sub_whisper=s_wh,
            del_whisper=d_wh,
            ins_whisper=i_wh,
        )
        if compute_cer:
            row.cer_dw = cer(ref_n, pdw_n)
            row.cer_donut = cer(ref_n, pdo_n)
            row.cer_whisper = cer(ref_n, pwh_n)
        rows.append(row)
    return rows, missing


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="包含 DW_small/ donut_only_zh/ whisper_only_zh/ 的目录",
    )
    ap.add_argument("--dw_json", type=str, default="DW_small/DW_small.json")
    ap.add_argument("--donut_json", type=str, default="donut_only_zh/donut_only_zh.json")
    ap.add_argument("--whisper_json", type=str, default="whisper_only_zh/whisper_only_zh.json")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/compare_dw_vs_baselines",
    )

    ap.add_argument("--normalize", action="store_true", help="对 ref/pred 做轻量归一化再判断对错")
    ap.add_argument("--keep_spaces", action="store_true", help="normalize 时不去空白")
    ap.add_argument("--no_punc_norm", action="store_true", help="normalize 时不做中英文标点映射")
    ap.add_argument("--lower", action="store_true", help="normalize 时转小写（中文通常无影响）")
    ap.add_argument("--keep_outer_quotes", action="store_true", help="normalize 时不移除最外层引号")

    ap.add_argument("--compute_cer", action="store_true", help="额外计算字符级 CER（归一化后）")
    ap.add_argument("--topk", type=int, default=200, help="导出样例的最大条数（每个类别）")
    args = ap.parse_args()

    dw_path = os.path.join(args.results_dir, args.dw_json)
    donut_path = os.path.join(args.results_dir, args.donut_json)
    whisper_path = os.path.join(args.results_dir, args.whisper_json)

    dw = read_result_json(dw_path)
    donut = read_result_json(donut_path)
    whisper = read_result_json(whisper_path)

    norm_kwargs = dict(
        lower=bool(args.lower),
        remove_spaces=not bool(args.keep_spaces),
        normalize_punc=not bool(args.no_punc_norm),
        remove_surrounding_quotes=not bool(args.keep_outer_quotes),
    )

    joined, missing = join_by_id(
        dw, donut, whisper, do_normalize=bool(args.normalize), norm_kwargs=norm_kwargs, compute_cer=bool(args.compute_cer)
    )

    ensure_dir(args.out_dir)
    dump_json(os.path.join(args.out_dir, "missing_ids.json"), missing)

    # ==========================
    # 字符级统计（按 ref 字符）
    # ==========================
    # 8 种组合统计：对每个 ref 字符位置，统计三模型在该字符是否 match
    combo_counter_chars: Counter = Counter()
    # 四象限同样按 ref 字符位置统计
    quad_dw_vs_donut_chars: Counter = Counter()
    quad_dw_vs_whisper_chars: Counter = Counter()

    def four_quadrants(a_ok: bool, b_ok: bool) -> str:
        if a_ok and b_ok:
            return "both_correct"
        if a_ok and (not b_ok):
            return "dw_only_correct"
        if (not a_ok) and b_ok:
            return "baseline_only_correct"
        return "both_wrong"

    # 为了 combo/四象限，需要得到“每个 ref 字符位置是否 match”的布尔数组
    def per_ref_match_mask(ref_n: str, hyp_n: str) -> List[bool]:
        ref_chars = list(ref_n)
        ops = align_ops(ref_chars, list(hyp_n))
        mask: List[bool] = []
        for op, rch, _ in ops:
            if rch is None:
                # insertion 不对应 ref 字符位置
                continue
            mask.append(op == "match")
        # 理论上 mask 长度应等于 len(ref_chars)
        if len(mask) != len(ref_chars):
            # 极端情况下回溯可能产生异常路径，这里兜底裁剪/补齐
            if len(mask) > len(ref_chars):
                mask = mask[: len(ref_chars)]
            else:
                mask = mask + [False] * (len(ref_chars) - len(mask))
        return mask

    # 重新遍历一次，用与 join_by_id 相同的 normalize 规则生成 mask
    # （避免把 normalize 的 ref_n 丢掉；这里复算开销可接受）
    for sid in sorted(set(dw) & set(donut) & set(whisper)):
        ref_raw = str(dw[sid].get("ref", ""))
        pdw_raw = str(dw[sid].get("pred", ""))
        pdo_raw = str(donut[sid].get("pred", ""))
        pwh_raw = str(whisper[sid].get("pred", ""))

        if args.normalize:
            ref_n = normalize_text(ref_raw, **norm_kwargs)
            pdw_n = normalize_text(pdw_raw, **norm_kwargs)
            pdo_n = normalize_text(pdo_raw, **norm_kwargs)
            pwh_n = normalize_text(pwh_raw, **norm_kwargs)
        else:
            ref_n, pdw_n, pdo_n, pwh_n = ref_raw, pdw_raw, pdo_raw, pwh_raw

        dw_mask = per_ref_match_mask(ref_n, pdw_n)
        do_mask = per_ref_match_mask(ref_n, pdo_n)
        wh_mask = per_ref_match_mask(ref_n, pwh_n)

        for a, b, c in zip(dw_mask, do_mask, wh_mask):
            combo_counter_chars[(a, b, c)] += 1
            quad_dw_vs_donut_chars[four_quadrants(a, b)] += 1
            quad_dw_vs_whisper_chars[four_quadrants(a, c)] += 1

    combo_stats = [
        {"ok_dw": k[0], "ok_donut": k[1], "ok_whisper": k[2], "count_chars": v}
        for k, v in sorted(combo_counter_chars.items(), key=lambda x: (-x[1], x[0]))
    ]
    dump_json(os.path.join(args.out_dir, "combo_stats.json"), combo_stats)

    dump_json(os.path.join(args.out_dir, "dw_vs_donut_quadrants.json"), dict(quad_dw_vs_donut_chars))
    dump_json(os.path.join(args.out_dir, "dw_vs_whisper_quadrants.json"), dict(quad_dw_vs_whisper_chars))

    # 导出样例：最关心 improvement / regression
    # - improvement: DW 对 & baseline 错
    # - regression: DW 错 & baseline 对
    def sample_row(r: JoinedRow) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": r.sid,
            "ref": r.ref,
            "pred_dw": r.pred_dw,
            "pred_donut": r.pred_donut,
            "pred_whisper": r.pred_whisper,
            "ok_dw": r.ok_dw,
            "ok_donut": r.ok_donut,
            "ok_whisper": r.ok_whisper,
            "ref_len_chars": r.ref_len,
            "match_dw": r.match_dw,
            "sub_dw": r.sub_dw,
            "del_dw": r.del_dw,
            "ins_dw": r.ins_dw,
            "match_donut": r.match_donut,
            "sub_donut": r.sub_donut,
            "del_donut": r.del_donut,
            "ins_donut": r.ins_donut,
            "match_whisper": r.match_whisper,
            "sub_whisper": r.sub_whisper,
            "del_whisper": r.del_whisper,
            "ins_whisper": r.ins_whisper,
        }
        if args.compute_cer:
            d.update(
                {
                    "cer_dw": r.cer_dw,
                    "cer_donut": r.cer_donut,
                    "cer_whisper": r.cer_whisper,
                }
            )
        return d

    def take_topk(rows: List[JoinedRow], key_fn=None) -> List[JoinedRow]:
        if key_fn is None:
            return rows[: args.topk]
        return sorted(rows, key=key_fn, reverse=True)[: args.topk]

    # 以 CER 差值排序（如果算了 CER），否则按原顺序
    def cer_gain_dw_vs_donut(r: JoinedRow) -> float:
        if r.cer_dw is None or r.cer_donut is None:
            return 0.0
        return float(r.cer_donut - r.cer_dw)

    def cer_gain_dw_vs_whisper(r: JoinedRow) -> float:
        if r.cer_dw is None or r.cer_whisper is None:
            return 0.0
        return float(r.cer_whisper - r.cer_dw)

    imp_vs_donut = [r for r in joined if r.ok_dw and (not r.ok_donut)]
    reg_vs_donut = [r for r in joined if (not r.ok_dw) and r.ok_donut]
    imp_vs_whisper = [r for r in joined if r.ok_dw and (not r.ok_whisper)]
    reg_vs_whisper = [r for r in joined if (not r.ok_dw) and r.ok_whisper]

    if args.compute_cer:
        imp_vs_donut = take_topk(imp_vs_donut, key_fn=cer_gain_dw_vs_donut)
        reg_vs_donut = take_topk(reg_vs_donut, key_fn=lambda r: -cer_gain_dw_vs_donut(r))
        imp_vs_whisper = take_topk(imp_vs_whisper, key_fn=cer_gain_dw_vs_whisper)
        reg_vs_whisper = take_topk(reg_vs_whisper, key_fn=lambda r: -cer_gain_dw_vs_whisper(r))
    else:
        imp_vs_donut = imp_vs_donut[: args.topk]
        reg_vs_donut = reg_vs_donut[: args.topk]
        imp_vs_whisper = imp_vs_whisper[: args.topk]
        reg_vs_whisper = reg_vs_whisper[: args.topk]

    dump_json(os.path.join(args.out_dir, "samples_dw_improve_vs_donut.json"), [sample_row(r) for r in imp_vs_donut])
    dump_json(os.path.join(args.out_dir, "samples_dw_regress_vs_donut.json"), [sample_row(r) for r in reg_vs_donut])
    dump_json(os.path.join(args.out_dir, "samples_dw_improve_vs_whisper.json"), [sample_row(r) for r in imp_vs_whisper])
    dump_json(os.path.join(args.out_dir, "samples_dw_regress_vs_whisper.json"), [sample_row(r) for r in reg_vs_whisper])

    # 全量明细（便于你用 pandas 二次分析）
    detail_rows = [sample_row(r) for r in joined]
    dump_json(os.path.join(args.out_dir, "joined_detail.json"), detail_rows)

    fieldnames = [
        "id",
        "ref",
        "pred_dw",
        "pred_donut",
        "pred_whisper",
        "ok_dw",
        "ok_donut",
        "ok_whisper",
        "ref_len_chars",
        "match_dw",
        "sub_dw",
        "del_dw",
        "ins_dw",
        "match_donut",
        "sub_donut",
        "del_donut",
        "ins_donut",
        "match_whisper",
        "sub_whisper",
        "del_whisper",
        "ins_whisper",
    ]
    if args.compute_cer:
        fieldnames += ["cer_dw", "cer_donut", "cer_whisper"]
    dump_csv(os.path.join(args.out_dir, "joined_detail.csv"), detail_rows, fieldnames=fieldnames)

    # 控制台摘要（也写一份 json）
    total_sents = len(joined)
    total_ref_chars = sum(r.ref_len for r in joined)
    total_match_dw = sum(r.match_dw for r in joined)
    total_match_donut = sum(r.match_donut for r in joined)
    total_match_whisper = sum(r.match_whisper for r in joined)
    total_ops_dw = {
        "match": total_match_dw,
        "sub": sum(r.sub_dw for r in joined),
        "del": sum(r.del_dw for r in joined),
        "ins": sum(r.ins_dw for r in joined),
    }
    total_ops_donut = {
        "match": total_match_donut,
        "sub": sum(r.sub_donut for r in joined),
        "del": sum(r.del_donut for r in joined),
        "ins": sum(r.ins_donut for r in joined),
    }
    total_ops_whisper = {
        "match": total_match_whisper,
        "sub": sum(r.sub_whisper for r in joined),
        "del": sum(r.del_whisper for r in joined),
        "ins": sum(r.ins_whisper for r in joined),
    }

    summary = {
        "unit": "char_ref_based",
        "total_joined_sents": total_sents,
        "total_ref_chars": total_ref_chars,
        "missing_counts": {k: len(v) for k, v in missing.items()},
        # 字符级准确率：match / total_ref_chars
        "acc_dw": (total_match_dw / total_ref_chars) if total_ref_chars else 0.0,
        "acc_donut": (total_match_donut / total_ref_chars) if total_ref_chars else 0.0,
        "acc_whisper": (total_match_whisper / total_ref_chars) if total_ref_chars else 0.0,
        "ops_dw": total_ops_dw,
        "ops_donut": total_ops_donut,
        "ops_whisper": total_ops_whisper,
        "dw_vs_donut_quadrants": dict(quad_dw_vs_donut_chars),
        "dw_vs_whisper_quadrants": dict(quad_dw_vs_whisper_chars),
        "combo_stats": combo_stats,
        "normalize": bool(args.normalize),
        "normalize_kwargs": norm_kwargs if args.normalize else None,
        "compute_cer": bool(args.compute_cer),
    }
    dump_json(os.path.join(args.out_dir, "summary.json"), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n已输出到: {args.out_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
产物：
  analysis/summary/runs_flat.csv
  analysis/summary/summary_ci.csv
  analysis/summary/paired_top2.csv
  analysis/summary/{species}_protocol_bars.png
"""

import os, re, glob, argparse, warnings
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -----------------------
# 路径推断
# -----------------------
def repo_root_from_here() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

def infer_ann_path(species: str, tag: str, repo_root: Optional[str] = None) -> Optional[str]:

    root = repo_root or repo_root_from_here()
    data_root = os.path.join(root, "data_process")
    if species in ("horse", "sheep"):
        for p in [
            os.path.join(data_root, species, "annotations", "test_annotations.coco.json"),
            os.path.join(data_root, species, "annotations", "test.json"),
        ]:
            if os.path.isfile(p): return p
        return None
    if species == "cattle":
        fold = "p1A" if "p1A" in tag else ("p1B" if "p1B" in tag else ("p1C" if "p1C" in tag else None))
        if fold is None:
            warnings.warn(f"[WARN] cattle tag 不含 p1A/p1B/p1C：{tag}，默认 p1A")
            fold = "p1A"
        for p in [
            os.path.join(data_root, "cattle", "annotations_p1", f"cattle_{fold}_test.json"),
            os.path.join(data_root, "cattle", "annotations_p1", f"cattle_{fold}_test_canon.json"),
        ]:
            if os.path.isfile(p): return p
        return None
    return None

FIXED_SIGMAS = np.asarray([
    0.015, 0.014, 0.020, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045,
    0.050, 0.051, 0.050, 0.051, 0.045, 0.050, 0.051, 0.050, 0.051
], dtype=np.float32)

def load_sigmas(K: int) -> np.ndarray:

    if K == FIXED_SIGMAS.size:
        return FIXED_SIGMAS.copy()
    if K < FIXED_SIGMAS.size:
        return FIXED_SIGMAS[:K].copy()
    return np.resize(FIXED_SIGMAS, K).astype(np.float32)

# -----------------------
# 评估 & 统计
# -----------------------
def eval_coco_ap(pred_json: str, ann_json: str) -> Dict[str,Any]:
    coco_gt = COCO(ann_json)
    coco_dt = coco_gt.loadRes(pred_json)


    k_gt = 0
    if "categories" in coco_gt.dataset and coco_gt.dataset["categories"]:
        kp = coco_gt.dataset["categories"][0].get("keypoints", [])
        if isinstance(kp, list) and kp:
            k_gt = len(kp)
    if k_gt <= 0:
        ann_ids = coco_gt.getAnnIds()
        if ann_ids:
            k_gt = len(coco_gt.loadAnns(ann_ids[0])[0].get("keypoints", [])) // 3
    if k_gt <= 0:
        raise RuntimeError("无法从 GT 推断关键点数")

    sigmas = load_sigmas(k_gt)

    E = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    E.params.kpt_oks_sigmas = sigmas
    E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    # s[0]=AP(0.5:0.95), s[1]=AP50, s[2]=AP75, s[5]=AR
    return dict(AP=float(s[0]), AP50=float(s[1]), AP75=float(s[2]), AR=float(s[5]),
                n_images=len(coco_gt.getImgIds()))

def mean_ci_95(arr: np.ndarray) -> Tuple[float,float,float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (np.nan, np.nan, np.nan)
    m = float(np.mean(arr))
    if arr.size < 2: return (m, np.nan, np.nan)
    s = float(np.std(arr, ddof=1))
    ci = 1.96 * s / np.sqrt(arr.size)
    return (m, m - ci, m + ci)

def bootstrap_diff_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 123) -> Tuple[float,float,float,float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    if x.size == 1:
        return (m, np.nan, np.nan, float(2.0 * (1.0 if m <= 0 else 0.0)))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2.0 * min(float(np.mean(boots <= 0.0)), float(np.mean(boots >= 0.0)))
    p = max(0.0, min(1.0, p))
    return (m, float(lo), float(hi), p)

def tag_to_method(tag: str) -> str:
    tl = tag.lower()
    if tl.startswith("yolox"): return "YOLOX"
    if tl.startswith("sar"):   return "SAR"
    return "SimCC"

def get_fold(species: str, tag: str) -> str:
    if species != "cattle": return "std"
    if "p1A" in tag: return "p1A"
    if "p1B" in tag: return "p1B"
    if "p1C" in tag: return "p1C"
    return "p1A"

# -----------------------
# 主流程
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dirs", nargs="+", default=[
        "analysis/raw_train_cattle",
        "analysis/raw_train_hs",
        "analysis/raw_smoke_cattle",
        "analysis/raw_smoke_hs",
    ])
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--out-dir", default="analysis/summary")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    repo_root = args.repo_root or repo_root_from_here()
    os.makedirs(args.out_dir, exist_ok=True)

    # 搜索预测
    preds = []
    for d in args.pred_dirs:
        if os.path.isdir(d):
            preds += glob.glob(os.path.join(d, "*.keypoints.json"))
    preds = sorted(set(preds))
    if not preds:
        raise SystemExit("[ERR] 没有找到 *.keypoints.json")

    pat = re.compile(r".*/(cattle|horse|sheep)_(.+)_s(\d+)\.keypoints\.json$")
    rows = []
    for p in preds:
        m = pat.match(p)
        if not m:
            warnings.warn(f"[SKIP] 名称不匹配：{p}")
            continue
        species, tag, seed = m.group(1), m.group(2), int(m.group(3))
        ann = infer_ann_path(species, tag, repo_root)
        if not ann or not os.path.isfile(ann):
            warnings.warn(f"[WARN] 未找到 GT ann：{species}/{tag} -> {ann}")
            continue

        try:
            res = eval_coco_ap(p, ann)
        except Exception as e:
            warnings.warn(f"[WARN] 评估失败：{p}\n原因：{e}")
            continue

        rows.append({
            "species": species,
            "tag": tag,
            "method": tag_to_method(tag),
            "fold": get_fold(species, tag),
            "seed": seed,
            **res,
            "pred_path": p,
            "ann_path": ann,
        })

    if not rows:
        raise SystemExit("[ERR] 没有成功评估的样本：请检查文件命名、GT 路径与 sigmas。")

    df = pd.DataFrame(rows).reset_index(drop=True)
    # 只保留 AP 有效的
    df = df.dropna(subset=["AP"]).reset_index(drop=True)

    # ---- 1) 每次 run 明细
    out_runs = os.path.join(args.out_dir, "runs_flat.csv")
    df.to_csv(out_runs, index=False)
    print(f"[OK] {out_runs}  ({len(df)} 行)")

    # ---- 2) 聚合（mean±95% CI）
    groups = []
    for (sp, tg, fd), g in df.groupby(["species","tag","fold"]):
        ap_m, ap_l, ap_u     = mean_ci_95(g["AP"].to_numpy())
        ap50_m, ap50_l, ap50_u = mean_ci_95(g["AP50"].to_numpy()) if "AP50" in g else (np.nan,np.nan,np.nan)
        ap75_m, ap75_l, ap75_u = mean_ci_95(g["AP75"].to_numpy()) if "AP75" in g else (np.nan,np.nan,np.nan)
        ar_m, ar_l, ar_u     = mean_ci_95(g["AR"].to_numpy()) if "AR" in g else (np.nan,np.nan,np.nan)
        groups.append({
            "species": sp, "fold": fd, "tag": tg, "method": tag_to_method(tg),
            "n_seeds": len(g),
            "AP_mean": ap_m, "AP_CI_lo": ap_l, "AP_CI_hi": ap_u,
            "AP50_mean": ap50_m, "AP50_CI_lo": ap50_l, "AP50_CI_hi": ap50_u,
            "AP75_mean": ap75_m, "AP75_CI_lo": ap75_l, "AP75_CI_hi": ap75_u,
            "AR_mean": ar_m, "AR_CI_lo": ar_l, "AR_CI_hi": ar_u,
        })
    agg = pd.DataFrame(groups).sort_values(
        ["species","fold","AP_mean"], ascending=[True,True,False]
    ).reset_index(drop=True)
    out_ci = os.path.join(args.out_dir, "summary_ci.csv")
    agg.to_csv(out_ci, index=False)
    print(f"[OK] {out_ci}")

    # ---- 3) 每个 species×fold 的 top-2 配对比较（bootstrap）
    paired_rows = []
    for (sp, fd), sub in agg.groupby(["species","fold"]):
        sub = sub.sort_values("AP_mean", ascending=False).reset_index(drop=True)
        if len(sub) < 2:
            continue
        top1, top2 = sub.iloc[0], sub.iloc[1]
        # 找到两个方法在 df 中共有的 seeds
        s1 = df[(df.species==sp)&(df.fold==fd)&(df.tag==top1["tag"])]
        s2 = df[(df.species==sp)&(df.fold==fd)&(df.tag==top2["tag"])]
        common = sorted(set(s1["seed"]).intersection(set(s2["seed"])))
        if not common:
            continue
        d = []
        for sd in common:
            a = float(s1[s1.seed==sd]["AP"].iloc[0])
            b = float(s2[s2.seed==sd]["AP"].iloc[0])
            d.append(a-b)
        d = np.asarray(d, dtype=float)
        mean_d, lo_d, hi_d, p = bootstrap_diff_ci(d, n_boot=args.n_boot, seed=123)
        paired_rows.append({
            "species": sp, "fold": fd,
            "first_tag": top1["tag"], "second_tag": top2["tag"],
            "first_AP_mean": float(top1["AP_mean"]), "second_AP_mean": float(top2["AP_mean"]),
            "diff_mean": mean_d, "diff_CI_lo": lo_d, "diff_CI_hi": hi_d,
            "boot_p_two_sided": p, "n_common_seeds": len(common)
        })
    paired = pd.DataFrame(paired_rows).sort_values(["species","fold"]).reset_index(drop=True)
    out_pair = os.path.join(args.out_dir, "paired_top2.csv")
    paired.to_csv(out_pair, index=False)
    print(f"[OK] {out_pair}")

    # ---- 4) 画跨协议柱状图（每个 species 一张；误差棒 NaN 保护）
    SERIES_COL = {"SimCC":"#4477AA", "SAR":"#66AA55", "YOLOX":"#CC6677"}
    for sp, g in agg.groupby("species"):
        folds = list(g["fold"].unique()); folds.sort()
        tags  = list(g["tag"].unique());  tags.sort()
        bar_w = 0.12
        x = np.arange(len(folds), dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, 1.6*len(folds)*len(tags)), 4.5))
        for i, tg in enumerate(tags):
            yy, yerr, col = [], [], SERIES_COL.get(tag_to_method(tg), "#888888")
            for fd in folds:
                row = g[(g["fold"]==fd)&(g["tag"]==tg)]
                if len(row)==0:
                    yy.append(np.nan); yerr.append(0.0)
                else:
                    r = row.iloc[0]
                    m  = float(r["AP_mean"])
                    lo = float(r["AP_CI_lo"]); hi = float(r["AP_CI_hi"])
                    err = m - lo if np.isfinite(lo) else (hi - m if np.isfinite(hi) else 0.0)
                    if not np.isfinite(err): err = 0.0
                    yy.append(m); yerr.append(err)
            pos = x + (i - (len(tags)-1)/2.0)*bar_w
            ax.bar(pos, yy, width=bar_w, label=tg, yerr=yerr, capsize=3,
                   color=col, edgecolor="white", linewidth=0.7, alpha=0.95)

        ax.set_xticks(x); ax.set_xticklabels(folds)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("AP (OKS .5:.95)")
        ax.set_title(f"{sp}: AP across protocols")
        ax.legend(ncol=min(len(tags), 4), fontsize=8, frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, f"{sp}_protocol_bars.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[OK] {out_png}")

if __name__ == "__main__":
    main()

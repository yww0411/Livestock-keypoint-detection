#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visibility slices evaluation (fixed 18-keypoint naming for all species):
- overall / visible / occluded (per-method, per-seed)
- per-joint AP for visible & occluded (only that joint contributes to OKS)
- fold(cattle)

Outputs:
  analysis/summary/vis_detail.csv   # per-seed details；含 kpt_id/kpt_name + 诊断
  analysis/summary/vis_summary.csv  # 聚合 mean ± 95% CI（overall/visible/occluded）
"""

import os, re, glob, argparse, warnings, copy, json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

KPT_NAMES_18 = [
    "mouth", "eye", "ear", "neck", "shoulder", "chest", "hip", "tail",
    "elbow", "left fore wrist", "left fore foot", "right fore wrist",
    "right fore foot", "hind knee", "left hind hock", "left hind foot",
    "right hind hock", "right hind foot"
]
K_EXPECT = 18

USER_SIGMAS_18 = np.asarray(
    [0.015, 0.014, 0.020, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045,
     0.050, 0.051, 0.050, 0.051, 0.045, 0.050, 0.051, 0.050, 0.051],
    dtype=np.float32
)

# -----------------------
# 路径
# -----------------------
def repo_root_from_here() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

def infer_ann_path(species: str, fold: str, repo_root: Optional[str] = None) -> Optional[str]:

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
        for p in [
            os.path.join(data_root, "cattle", "annotations_p1", f"cattle_{fold}_test_canon.json"),
            os.path.join(data_root, "cattle", "annotations_p1", f"cattle_{fold}_test.json"),
        ]:
            if os.path.isfile(p): return p
        return None
    return None

# -----------------------
# sigmas
# -----------------------
def robust_load_sigmas(k_gt: int) -> np.ndarray:
    if k_gt != K_EXPECT:
        warnings.warn(f"[WARN] GT 关键点数 {k_gt} != {K_EXPECT}，将按 K 裁剪/重复 sigmas 与名称。")
    if k_gt <= len(USER_SIGMAS_18):
        sig = USER_SIGMAS_18[:k_gt].copy()
    else:
        sig = np.resize(USER_SIGMAS_18, k_gt).copy()
    return sig.astype(np.float32)

def get_fixed_kpt_names(k_gt: int) -> List[str]:
    if k_gt == K_EXPECT:
        return KPT_NAMES_18[:]
    if k_gt <= len(KPT_NAMES_18):
        return KPT_NAMES_18[:k_gt]
    else:
        extra = [f"J{i+1:02d}" for i in range(len(KPT_NAMES_18), k_gt)]
        return KPT_NAMES_18 + extra

# -----------------------
# 牛的三份 GT 预加载（用于交集判定 fold）
# -----------------------
_CATTLE_GT_CACHE = None

def preload_cattle_gt_sets(repo_root: Optional[str]) -> Dict[str, Dict[str, Any]]:

    global _CATTLE_GT_CACHE
    if _CATTLE_GT_CACHE is not None:
        return _CATTLE_GT_CACHE

    root = repo_root or repo_root_from_here()
    base = os.path.join(root, "data_process", "cattle", "annotations_p1")
    out: Dict[str, Dict[str, Any]] = {}
    for fd in ["p1A", "p1B", "p1C"]:
        ann = None
        for name in [f"cattle_{fd}_test_canon.json", f"cattle_{fd}_test.json"]:
            p = os.path.join(base, name)
            if os.path.isfile(p):
                ann = p
                break
        if not ann:
            continue
        coco = COCO(ann)
        out[fd] = {"ann": ann, "img_ids": set(int(i) for i in coco.getImgIds())}
    _CATTLE_GT_CACHE = out
    return out

def detect_fold_by_overlap(species: str, pred_filepath: str, repo_root: Optional[str]) -> str:

    if str(species).lower() != "cattle":
        return "std"

    m = re.search(r"\bp1([abc])\b", pred_filepath, flags=re.IGNORECASE)
    if m:
        return f"p1{m.group(1).upper()}"


    try:
        with open(pred_filepath, "r", encoding="utf-8") as f:
            dts = json.load(f)
        pred_ids = set(int(d.get("image_id", -1)) for d in dts if "image_id" in d)
        pred_ids.discard(-1)
    except Exception:
        pred_ids = set()

    gt_sets = preload_cattle_gt_sets(repo_root)
    best_fold, best_overlap = "p1A", -1
    for fd, v in gt_sets.items():
        inter = len(pred_ids & v["img_ids"])
        if inter > best_overlap:
            best_overlap = inter
            best_fold = fd

    if best_overlap <= 0:
        print(f"[WARN] cannot infer fold by overlap, default to p1A | file={os.path.basename(pred_filepath)}")
    else:
        print(f"[FOLD] {os.path.basename(pred_filepath)} -> {best_fold} (overlap={best_overlap})")
    return best_fold

# -----------------------
# 构造“点级子集”的 COCO GT，并返回保留 image_id
# -----------------------
def build_subset_gt(ann_json: str, subset: str, joint_idx: Optional[int]=None) -> Tuple[COCO, List[int]]:
    with open(ann_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    data2 = copy.deepcopy(data)
    anns_new: List[Dict[str, Any]] = []
    keep_img_ids: set = set()

    for ann in data2.get("annotations", []):
        kps = ann.get("keypoints", [])
        if not (isinstance(kps, list) and len(kps) % 3 == 0):
            continue
        K = len(kps) // 3
        v = np.asarray(kps[2::3], dtype=np.int32)

        if subset == "overall":
            keep = (v > 0)
        elif subset == "visible":
            keep = (v == 2)
        elif subset == "occluded":
            keep = (v == 1)
        else:
            keep = (v > 0)

        if joint_idx is not None and 0 <= joint_idx < K:
            only = np.zeros(K, dtype=bool); only[joint_idx] = True
            keep = keep & only

        if keep.any():
            new_kps = kps[:]
            for i in range(K):
                new_kps[3*i + 2] = 2 if keep[i] else 0
            ann2 = dict(ann)
            ann2["keypoints"] = new_kps
            ann2["num_keypoints"] = int(keep.sum())
            anns_new.append(ann2)
            keep_img_ids.add(ann["image_id"])

    imgs_new = [im for im in data2.get("images", []) if im["id"] in keep_img_ids]
    data2["annotations"] = anns_new
    data2["images"] = imgs_new

    coco = COCO()
    coco.dataset = data2
    coco.createIndex()
    return coco, sorted(keep_img_ids)

# -----------------------
# 预测健壮装载 + 过滤到 kept_img_ids
# -----------------------
def loadRes_filtered(coco_gt: COCO, pred_json: str, keep_img_ids: List[int]):

    keep = set(int(i) for i in keep_img_ids)

    # 从 GT 推断 K & 类别 id
    gt_K = 0
    ann_ids = coco_gt.getAnnIds()
    if ann_ids:
        any_ann = coco_gt.loadAnns([ann_ids[0]])[0]
        gt_K = len(any_ann.get("keypoints", [])) // 3
    if gt_K <= 0:
        cats = coco_gt.dataset.get("categories", [])
        if cats and "keypoints" in cats[0]:
            gt_K = len(cats[0]["keypoints"])
    if gt_K <= 0:
        raise RuntimeError("Cannot infer K from coco_gt")

    cat_ids = coco_gt.getCatIds()
    gt_cat_id = int(cat_ids[0]) if cat_ids else 1

    # 读预测
    with open(pred_json, "r", encoding="utf-8") as f:
        dts_all = json.load(f)

    bad_len = bad_type = bad_img = 0
    fixed_pad = fixed_trunc = filled_score = filled_cat = 0

    def norm_kps(kps):
        nonlocal fixed_pad, fixed_trunc, bad_type
        if not isinstance(kps, list):
            bad_type += 1; return None
        try:
            kps = [float(x) for x in kps]
        except Exception:
            bad_type += 1; return None
        # 长度不是 3 的倍数，先截断到最近的 3*n
        if len(kps) % 3 != 0:
            kps = kps[: (len(kps)//3)*3]
        dtK = len(kps)//3
        if dtK == gt_K:
            return kps
        if dtK > gt_K:
            fixed_trunc += 1
            return kps[:3*gt_K]
        need = gt_K - dtK
        fixed_pad += 1
        return kps + [0.0, 0.0, 0.0]*need

    dts = []
    for d in dts_all:
        try:
            img_id = int(d.get("image_id", -1))
        except Exception:
            bad_type += 1
            continue
        if img_id not in keep:
            bad_img += 1
            continue

        kps = d.get("keypoints", [])
        kps = norm_kps(kps)
        if kps is None or len(kps) != 3*gt_K:
            bad_len += 1
            continue

        dd = dict(d)
        dd["image_id"] = img_id
        dd["keypoints"] = kps
        if "score" not in dd:
            dd["score"] = 1.0; filled_score += 1
        if "category_id" not in dd:
            dd["category_id"] = gt_cat_id; filled_cat += 1
        dts.append(dd)

    if bad_len or bad_type or bad_img or fixed_pad or fixed_trunc or filled_score or filled_cat:
        print(f"[SANITIZE] {os.path.basename(pred_json)} | kept={len(dts)} | "
              f"pad={fixed_pad} trunc={fixed_trunc} fill_sc={filled_score} fill_cat={filled_cat} | "
              f"drop_bad_len={bad_len} drop_bad_type={bad_type} drop_out_img={bad_img}")

    return coco_gt.loadRes(dts)

# -----------------------
# 评估
# -----------------------
def coco_eval(pred_json: str, coco_gt: COCO, keep_img_ids: List[int], sigmas: np.ndarray) -> Dict[str, float]:
    if not coco_gt.getAnnIds():
        return dict(AP=np.nan, AP50=np.nan, AP75=np.nan)
    coco_dt = loadRes_filtered(coco_gt, pred_json, keep_img_ids)
    if len(coco_dt.getAnnIds()) == 0:
        return dict(AP=np.nan, AP50=np.nan, AP75=np.nan)
    E = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    E.params.kpt_oks_sigmas = sigmas
    E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return dict(AP=float(s[0]), AP50=float(s[1]), AP75=float(s[2]))

def mean_ci_95(arr: np.ndarray) -> Tuple[float, float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (np.nan, np.nan, np.nan)
    m = float(np.mean(arr))
    if arr.size < 2: return (m, np.nan, np.nan)
    s = float(np.std(arr, ddof=1))
    ci = 1.96 * s / np.sqrt(arr.size)
    return (m, m - ci, m + ci)

def tag_to_method(tag: str) -> str:
    t = str(tag).lower()
    if t.startswith("yolox"): return "yolox"
    if t.startswith("sar"):   return "sar"
    return "simcc"

# -----------------------
# 诊断
# -----------------------
def subset_stats_from_coco(coco_gt: COCO) -> Tuple[int, int, int]:
    ann_ids = coco_gt.getAnnIds()
    anns = coco_gt.loadAnns(ann_ids)
    n_inst = len(anns)
    n_img = len(coco_gt.getImgIds())
    n_kpts = 0
    for a in anns:
        kps = a.get("keypoints", [])
        if kps:
            n_kpts += sum(1 for vv in kps[2::3] if vv > 0)
    return n_img, n_inst, n_kpts

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
    ap.add_argument("--out-dir", default="analysis/summary")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args()

    repo_root = args.repo_root or repo_root_from_here()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 搜索预测
    preds = []
    for d in args.pred_dirs:
        preds += glob.glob(os.path.join(d, "*.keypoints.json"))
    preds = sorted(set(preds))
    print(f"[INFO] 发现预测文件数: {len(preds)}")
    if not preds:
        raise SystemExit("[ERR] 没有 *.keypoints.json")


    pat = re.compile(
        r".*/(cattle|horse|sheep)_(.+?)[-_](?:s|seed)(\d+)\.keypoints\.json$",
        re.IGNORECASE
    )

    detail_rows: List[Dict[str, Any]] = []

    for p in preds:
        m = pat.match(p)
        if not m:
            warnings.warn(f"[SKIP] 名称不匹配：{p}")
            continue
        species = m.group(1).lower()
        tag = m.group(2)
        seed = int(m.group(3))

        # 关键：用交集自动判定 fold（horse/sheep 恒为 std）
        fold = detect_fold_by_overlap(species, p, repo_root)
        ann = infer_ann_path(species, fold, repo_root)
        if not ann or not os.path.isfile(ann):
            warnings.warn(f"[WARN] 未找到 GT：{species}/{fold} <- {ann}")
            continue

        # 推断 K
        coco_tmp = COCO(ann)
        K = 0
        if "categories" in coco_tmp.dataset and coco_tmp.dataset["categories"]:
            kp = coco_tmp.dataset["categories"][0].get("keypoints", [])
            if isinstance(kp, list) and kp: K = len(kp)
        if K <= 0:
            ann_ids = coco_tmp.getAnnIds()
            if ann_ids:
                K = len(coco_tmp.loadAnns(ann_ids[0])[0].get("keypoints", [])) // 3
        if K <= 0:
            warnings.warn(f"[WARN] 无法推断关键点数，跳过 {p}")
            continue

        # 固定 sigmas & 名字
        sigmas = robust_load_sigmas(K)
        kpt_names = get_fixed_kpt_names(K)

        # ---- overall / visible / occluded（全体关节）
        for subset in ("overall", "visible", "occluded"):
            coco_gt, keep_img_ids = build_subset_gt(ann, subset, joint_idx=None)
            try:
                res = coco_eval(p, coco_gt, keep_img_ids, sigmas)
                n_img, n_inst, n_kpts = subset_stats_from_coco(coco_gt)
            except Exception as e:
                warnings.warn(f"[WARN] eval 失败（{subset}）：{p}\n{e}")
                continue
            detail_rows.append({
                "species": species, "fold": fold, "tag": tag, "method": tag_to_method(tag),
                "seed": seed, "subset": subset,
                "AP": res["AP"], "AP50": res["AP50"], "AP75": res["AP75"],
                "kpt_id": -1, "kpt_name": "(all)",
                "n_images": n_img, "n_instances": n_inst, "n_kpts": n_kpts
            })

        # ---- 每个关节：visible / occluded（只该关节参与 OKS）
        for j in range(K):
            for subset in ("visible", "occluded"):
                coco_gt, keep_img_ids = build_subset_gt(ann, subset, joint_idx=j)
                try:
                    res = coco_eval(p, coco_gt, keep_img_ids, sigmas)
                    n_img, n_inst, n_kpts = subset_stats_from_coco(coco_gt)
                except Exception as e:
                    warnings.warn(f"[WARN] eval 失败（joint {j+1} {subset}）：{p}\n{e}")
                    continue
                detail_rows.append({
                    "species": species, "fold": fold, "tag": tag, "method": tag_to_method(tag),
                    "seed": seed, "subset": subset,
                    "AP": res["AP"], "AP50": res["AP50"], "AP75": res["AP75"],
                    "kpt_id": j + 1, "kpt_name": kpt_names[j],
                    "n_images": n_img, "n_instances": n_inst, "n_kpts": n_kpts
                })

    if not detail_rows:
        raise SystemExit("[ERR] 没有成功评估的样本（detail 为空）。请检查预测命名、GT 路径与 sigmas。")


    df_detail = pd.DataFrame(detail_rows).sort_values(
        ["species", "fold", "method", "tag", "seed", "kpt_id", "subset"]
    ).reset_index(drop=True)
    out_detail = out_dir / "vis_detail.csv"
    df_detail.to_csv(out_detail, index=False)
    print(f"[OK] {out_detail}  ({len(df_detail)} 行)")

    tmp = df_detail.copy()
    print("\n[DIAG] overall rows per (species, fold, subset):")
    print(tmp[tmp["kpt_id"] < 0].groupby(["species","fold","subset"]).size().unstack("subset", fill_value=0))

    print("\n[DIAG] #joints with non-empty slices (per species×fold):")
    print(tmp[tmp["kpt_id"] > 0].groupby(["species","fold","subset"]).size().unstack("subset", fill_value=0))

    # ---- 汇总到 vis_summary（只统计整体：kpt_id == -1）
    base = df_detail[df_detail["kpt_id"] < 0].copy()
    groups = []
    for (sp, fd, md, sb), g in base.groupby(["species", "fold", "method", "subset"]):
        ap_m, ap_l, ap_u = mean_ci_95(g["AP"].to_numpy())
        groups.append({
            "species": sp, "fold": fd, "method": md, "subset": sb,
            "AP_mean": ap_m, "AP_CI_lo": ap_l, "AP_CI_hi": ap_u,
            "n_seeds": len(g["seed"].unique()),
            "n_images": int(g["n_images"].sum()),
            "n_instances": int(g["n_instances"].sum()),
            "n_kpts": int(g["n_kpts"].sum()),
        })
    df_sum = pd.DataFrame(groups).sort_values(
        ["species", "fold", "method", "subset"]
    ).reset_index(drop=True)
    out_sum = out_dir / "vis_summary.csv"
    df_sum.to_csv(out_sum, index=False)
    print(f"[OK] {out_sum}")


if __name__ == "__main__":
    main()

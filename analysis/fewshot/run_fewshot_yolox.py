#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot fine-tune & eval for YOLOX-Pose on AP-10K sheep (42 imgs).

- Tags: yolox_s_std, yolox_m_std
- Shots: K in {0, 10, 20, 30}; Seeds: 1337, 2029, 3407
- K=0: zero-shot 仅评测；K>0: 微调后评测
- 预测统一写到 analysis/public_preds/ap10k_<tag>_k<K>_s<seed>.keypoints.json
"""

from __future__ import annotations
from pathlib import Path
import os, sys, json, random, shutil, subprocess, warnings
from typing import Optional, Any, Dict

# ---------------- repo paths ----------------
ROOT = Path(__file__).resolve().parents[2]   # repo root: hy-tmp/keypoint-detection

PY_PATHS = [
    str(ROOT),
    str(ROOT/'yolox_pose-main'),
    str(ROOT/'SimCC-main'),
    str(ROOT/'SAR-main'),
]
for p in PY_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ['PYTHONPATH'] = os.pathsep.join(PY_PATHS + [os.environ.get('PYTHONPATH', '')])

os.chdir(str(ROOT))

from mmengine.config import Config
from mmengine.registry import DefaultScope
try:
    from mmengine.registry import init_default_scope
except Exception:
    init_default_scope = None

AP10K_ROOT   = ROOT/"data_process"/"test_data"
AP10K_ANN    = AP10K_ROOT/"annotations"/"test_annotations.coco.json"
AP10K_IMAGES = AP10K_ROOT/"images"

def set_scope(name: str):
    try:
        cur = DefaultScope.get_current_instance()
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
    except Exception:
        pass
    if init_default_scope is not None:
        try:
            init_default_scope(name)  # >=0.7
            return
        except Exception:
            pass
    try:
        DefaultScope.get_instance(name, scope_name=name)
        return
    except Exception:
        pass
    try:
        DefaultScope(name=name, scope_name=name)
    except Exception:
        warnings.warn("[WARN] DefaultScope 设置失败，但继续。")

# ---------------- configs / tags ----------------
CFGS = {
    "yolox_s_std": str(ROOT/"yolox_pose-main/configs/yolox-pose_s_8xb32-300e_sheep_coco.py"),
    "yolox_m_std": str(ROOT/"yolox_pose-main/configs/yolox-pose_m_4xb16-300e_sheep_coco.py"),
}
TAGS  = list(CFGS.keys())
SEEDS = (1337, 2029, 3407)
SHOTS = (0, 10, 20, 30)

WORK_ROOT  = ROOT/"work_dirs_fewshot_yolox"
PRED_ROOT  = ROOT/"analysis/public_preds"
SPLIT_ROOT = ROOT/"analysis/fewshot_splits_ap10k"
for d in (WORK_ROOT, PRED_ROOT, SPLIT_ROOT):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- utilities ----------------
def find_ckpt(tag: str, seed: int) -> Optional[str]:
    wd = ROOT/"work_dirs_train_hs"/f"{tag}_sheep_s{seed}"
    if not wd.is_dir():
        return None
    cand = sorted(list(wd.glob("best*.pth")) + list(wd.glob("epoch_*.pth")))
    return str(cand[-1]) if cand else None

def read_coco(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def make_kshot_split(ann_in: Path, images_dir: Path, K: int, seed: int) -> Path:
    """从 AP-10K test_annotations.coco.json 里按图像采样 K 张，生成训练 JSON。"""
    out = SPLIT_ROOT/f"ap10k_k{K}_s{seed}.coco.json"
    if out.is_file():
        return out

    data = read_coco(ann_in)
    imgs = data.get("images", [])
    id2img = {im["id"]: im for im in imgs}

    # 所有“有关键点”的图像
    ann_by_img = {}
    for a in data.get("annotations", []):
        if a.get("iscrowd", 0):
            continue
        if "keypoints" not in a:
            continue
        ann_by_img.setdefault(a["image_id"], []).append(a)

    valid_img_ids = [i for i in ann_by_img.keys() if (images_dir / id2img[i]["file_name"]).is_file()]
    if len(valid_img_ids) < K:
        warnings.warn(f"[WARN] AP-10K 可用图像 {len(valid_img_ids)} < K={K}，将使用全部。")
        K = len(valid_img_ids)

    rng = random.Random(seed)
    pick_ids = sorted(rng.sample(valid_img_ids, K)) if K > 0 else []

    new_images, new_anns = [], []
    next_img_id, next_ann_id = 1, 1
    for old_id in pick_ids:
        im = id2img[old_id]
        im_new = im.copy(); im_new["id"] = next_img_id
        new_images.append(im_new)
        for a in ann_by_img.get(old_id, []):
            aa = a.copy()
            aa["id"] = next_ann_id
            aa["image_id"] = next_img_id
            if "num_keypoints" not in aa:
                kpts = aa.get("keypoints", [])
                aa["num_keypoints"] = int(sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0))
            new_anns.append(aa)
            next_ann_id += 1
        next_img_id += 1

    out_obj = {
        "images": new_images,
        "annotations": new_anns,
        "categories": data.get("categories", []),
        "licenses": data.get("licenses", []),
        "info": data.get("info", {}),
    }
    write_json(out_obj, out)
    return out

def _patch_evaluator(evaluator: Any, ann_file: str, outfile_prefix: str):
    if evaluator is None:
        return
    if isinstance(evaluator, list):
        for ev in evaluator:
            _patch_evaluator(ev, ann_file, outfile_prefix)
        return
    if isinstance(evaluator, dict):
        evaluator["ann_file"] = ann_file
        evaluator["outfile_prefix"] = outfile_prefix
        if "format_only" in evaluator:
            evaluator["format_only"] = True
        return

def build_cfg_for(tag: str, seed: int, K: int, ckpt: str, train_ann: Optional[Path]) -> Path:
    set_scope("mmyolo")
    cfg = Config.fromfile(CFGS[tag])

    # —— train dataloader
    if K > 0 and train_ann is not None and "train_dataloader" in cfg and "dataset" in cfg.train_dataloader:
        cfg.train_dataloader.dataset.ann_file = str(train_ann)
        cfg.train_dataloader.dataset.data_root = str(AP10K_ROOT)
        if hasattr(cfg.train_dataloader.dataset, "data_prefix"):
            dp = cfg.train_dataloader.dataset.data_prefix
            if isinstance(dp, dict):
                dp["img"] = "images"
        if "batch_size" in cfg.train_dataloader:
            cfg.train_dataloader.batch_size = min(16, max(4, int(cfg.train_dataloader.batch_size) // 2))
        if "num_workers" in cfg.train_dataloader:
            cfg.train_dataloader.num_workers = max(2, int(cfg.train_dataloader.num_workers) // 2)


    for key in ("val_dataloader", "test_dataloader"):
        if key in cfg and "dataset" in cfg[key]:
            cfg[key].dataset.ann_file = str(AP10K_ANN)
            cfg[key].dataset.data_root = str(AP10K_ROOT)
            if hasattr(cfg[key].dataset, "data_prefix"):
                dp = cfg[key].dataset.data_prefix
                if isinstance(dp, dict):
                    dp["img"] = "images"

    # evaluator：保存 keypoints JSON（支持 list/dict）
    out_prefix = str(PRED_ROOT/f"ap10k_{tag}_k{K}_s{seed}")
    for ev_key in ("val_evaluator", "test_evaluator"):
        if ev_key in cfg:
            _patch_evaluator(cfg[ev_key], str(AP10K_ANN), out_prefix)

    if K > 0:
        EPOCHS_FOR_K_TOPDOWN = {10: 30, 20: 40, 30: 60}
        target_epochs = EPOCHS_FOR_K_TOPDOWN.get(K, 30)


        cfg.train_cfg = dict(
            type='EpochBasedTrainLoop',
            max_epochs=target_epochs,
            val_interval=1,
        )

        if "optim_wrapper" in cfg and "optimizer" in cfg.optim_wrapper:
            base_lr = float(cfg.optim_wrapper.optimizer.get("lr", 0.001))
            cfg.optim_wrapper.optimizer["lr"] = max(1e-4, base_lr * 0.5)

        if "param_scheduler" in cfg:
            for sch in cfg.param_scheduler:
                if not isinstance(sch, dict) or "type" not in sch:
                    continue
                t = sch["type"]
                if "MultiStep" in t and "milestones" in sch:
                    sch["milestones"] = [max(2, int(0.7 * target_epochs))]
                    if "gamma" in sch:
                        sch["gamma"] = min(0.5, sch["gamma"])
                if "Cosine" in t and "T_max" in sch:
                    sch["T_max"] = target_epochs
                if "LinearLR" in t:
                    sch["start_factor"] = max(0.01, sch.get("start_factor", 0.1))
                    sch["by_epoch"] = True
                    sch["begin"] = 0
                    sch["end"] = max(1, int(0.1 * target_epochs))

        hooks = []
        if "custom_hooks" in cfg and isinstance(cfg.custom_hooks, (list, tuple)):
            hooks.extend(cfg.custom_hooks)
        if "default_hooks" in cfg and isinstance(cfg.default_hooks, dict):
            hooks.extend([cfg.default_hooks[k] for k in cfg.default_hooks if isinstance(cfg.default_hooks[k], dict)])
        for h in hooks:
            try:
                if h.get("type") in ("YOLOXModeSwitchHook", "PPYOLOEModeSwitchHook"):
                    h["num_last_epochs"] = max(2, int(0.3 * target_epochs))
            except Exception:
                pass


    # 日志/保存与载入
    tag_name = f"{tag}_ap10k_k{K}_s{seed}"
    work_dir = WORK_ROOT/tag_name
    cfg.work_dir = str(work_dir)
    cfg.load_from = ckpt
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = work_dir/"cfg_ap10k.py"
    cfg.dump(str(cfg_path))
    return cfg_path

def run_train(cfg_path: Path) -> bool:
    env = os.environ.copy()
    env['PYTHONPATH'] = os.environ['PYTHONPATH']
    cmd = [sys.executable, str(ROOT/"tools/train.py"), str(cfg_path)]
    print("[CMD]", " ".join(cmd))
    ret = subprocess.run(cmd, env=env)
    return ret.returncode == 0

def pick_best_ckpt(work_dir: Path) -> Optional[Path]:
    cands = list(work_dir.glob("best*.pth"))
    if not cands:
        cands = list(work_dir.glob("epoch_*.pth"))
    return sorted(cands)[-1] if cands else None

def run_test(cfg_path: Path, ckpt_path: Path, tag: str, K: int, seed: int) -> Optional[Path]:
    env = os.environ.copy()
    env['PYTHONPATH'] = os.environ['PYTHONPATH']
    cmd = [sys.executable, str(ROOT/"tools/test.py"), str(cfg_path), str(ckpt_path)]
    print("[CMD]", " ".join(cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        return None

    patt = f"ap10k_{tag}_k{K}_s{seed}*keypoints*.json"
    hits = sorted(PRED_ROOT.glob(patt))
    if hits:
        return hits[-1]

    patt2 = f"ap10k_{tag}_k{K}_s{seed}*.json"
    hits = sorted(PRED_ROOT.glob(patt2))
    if hits:
        return hits[-1]

    work_dir = Path(cfg_path).parent
    hits = sorted(work_dir.glob("*.keypoints*.json"))
    if hits:
        return hits[-1]

    hits = sorted(work_dir.glob("*.json"))
    return hits[-1] if hits else None

# ---------------- main ----------------
def main():
    assert AP10K_ANN.is_file(), f"missing AP-10K ann: {AP10K_ANN}"
    assert AP10K_IMAGES.is_dir(), f"missing AP-10K images dir: {AP10K_IMAGES}"

    for tag in TAGS:
        cfg_file = Path(CFGS[tag])
        if not cfg_file.is_file():
            warnings.warn(f"[SKIP] missing cfg: {cfg_file}")
            continue

        for seed in SEEDS:
            native = find_ckpt(tag, seed)
            if not native:
                warnings.warn(f"[SKIP] no native ckpt for {tag} s{seed}")
                continue

            for K in SHOTS:
                print(f"\n[RUN] {tag} | K={K} | s{seed}")
                train_ann = None
                if K > 0:
                    train_ann = make_kshot_split(AP10K_ANN, AP10K_IMAGES, K, seed)

                cfg_path = build_cfg_for(tag, seed, K, native, train_ann)
                work_dir = cfg_path.parent

                if K > 0:
                    ok = run_train(cfg_path)
                    if not ok:
                        warnings.warn(f"[WARN] train failed: {tag} K{K} s{seed}")
                        continue
                    ckpt = pick_best_ckpt(work_dir)
                    if not ckpt:
                        warnings.warn(f"[WARN] no ckpt after train: {work_dir}")
                        continue
                else:
                    ckpt = Path(native)

                out_json = run_test(cfg_path, ckpt, tag, K, seed)
                if out_json is not None:
                    dst = PRED_ROOT/f"ap10k_{tag}_k{K}_s{seed}.keypoints.json"
                    try:
                        shutil.copy(out_json, dst)
                    except Exception:
                        shutil.move(out_json, dst)
                    print("[OUT]", dst)
                else:
                    warnings.warn(f"[WARN] test finished but JSON not found: {tag} K{K} s{seed}")

if __name__ == "__main__":
    main()

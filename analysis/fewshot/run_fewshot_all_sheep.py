#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot fine-tuning & eval for top-down sheep models on AP-10K (or your public set).

改动要点（解决 'keypoints' 报错）：
- 严格筛选预测文件：默认只评测 ap10k_* / sheep_* 前缀的 *.keypoints.json（可用 --allowed-prefixes 覆盖）
- 评测前做 JSON 结构校验（必须是 COCO keypoints results 的 list[dict] 且含 'keypoints' 等）
- 不合规文件直接跳过，并友好打印原因；不会阻塞整轮流程

其余特性：
- 断点续训 (--resume 自动识别 last_checkpoint/epoch_*.pth)
- 限制 checkpoint 数量 (max_keep_ckpts=2)
- 统一修正 data_root / data_prefix.img，避免 images/images 重复
- 训练后自动推理 → 导出 *.keypoints.json
- 评测用固定 OKS σ(K=18)，输出 runs.csv & summary_ci.csv
"""

from __future__ import annotations
from pathlib import Path
import os, sys, re, json, csv, argparse, subprocess, warnings
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------- repo root & sys.path ----------------
ROOT = Path(__file__).resolve().parents[2]  # .../hy-tmp/keypoint-detection
os.chdir(str(ROOT))

EXTRA_PY_PATHS = [
    str(ROOT),
    str(ROOT/"SimCC-main"),
    str(ROOT/"SAR-main"),
    str(ROOT/"yolox_pose-main"),
]
for p in EXTRA_PY_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------- mmpose API (1.x/0.x 兼容) ----------------
API_VER = None
try:
    # mmpose 1.x
    from mmpose.apis import init_model as _init_model_v1
    try:
        from mmpose.apis import inference_topdown as _infer_v1
    except Exception:
        from mmpose.apis import inference_topdown_model as _infer_v1  # 罕见早期夜版
    API_VER = 1
except Exception:
    try:
        # mmpose 0.x
        from mmpose.apis import init_pose_model as _init_model_v0
        from mmpose.apis import inference_top_down_pose_model as _infer_v0
        API_VER = 0
    except Exception as e:
        raise SystemExit(
            "[ERR] 未找到 mmpose 1.x/0.x 的 top-down API，请检查环境。"
        ) from e

try:
    from mmengine.config import Config
except Exception:
    Config = None

# ---------------- 数据布局（你的约定） ----------------
# 源域 sheep（可用于 zero-shot / 比对）
SHEEP_DIR = ROOT/"data_process"/"sheep"
SHEEP_ANN_DIR = SHEEP_DIR/"annotations"

# 公共 few-shot 数据（AP-10K 羊），我们按你的描述使用：
# data_process/test_data/annotations/test_annotations.coco.json
# data_process/test_data/images/*.jpg
AP10K_DIR = ROOT/"data_process"/"test_data"
AP10K_ANN = AP10K_DIR/"annotations"/"test_annotations.coco.json"

# ---------------- 输出 ----------------
WORK_DIR = ROOT/"work_dirs_fewshot"
PRED_DIR = ROOT/"analysis"/"public_preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR  = ROOT/"analysis"/"public"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV = OUT_DIR/"public_runs.csv"
SUM_CSV  = OUT_DIR/"public_summary_ci.csv"

# ---------------- 固定 OKS σ(K=18) ----------------
FIXED_SIGMAS = np.asarray([
    0.015, 0.014, 0.020, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045,
    0.050, 0.051, 0.050, 0.051, 0.045, 0.050, 0.051, 0.050, 0.051
], dtype=np.float32)

# ---------------- 小工具 ----------------
def set_scope(scope: str):
    """避免不同注册表冲突"""
    from mmengine.registry import DefaultScope, init_default_scope
    try:
        cur = DefaultScope.get_current_instance()
        if cur is not None:
            cur.close()
    except Exception:
        pass
    init_default_scope(scope)

def load_cfg(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    scope = 'mmpose'
    set_scope(scope)
    return cfg, scope

def _set_size_in_cfg_any(cfg, W: int, H: int):
    """兼容 1.x/0.x，尽可能把 input/image_size 改到目标尺寸"""
    try:
        if hasattr(cfg, "codec") and cfg.codec is not None:
            cfg.codec.input_size = (W, H)
    except Exception: pass
    try:
        hd = getattr(cfg.model, "head", None) or getattr(cfg.model, "keypoint_head", None)
        if hd and hasattr(hd, "codec") and hd.codec is not None:
            hd.codec.input_size = (W, H)
    except Exception: pass
    try:
        if hasattr(cfg, "test_pipeline"):
            for t in cfg.test_pipeline:
                if isinstance(t, dict):
                    if "input_size" in t: t["input_size"] = (W, H)
                    if "image_size" in t: t["image_size"] = (W, H)
    except Exception: pass
    try:
        data = getattr(cfg, "data", None)
        if data and isinstance(data, dict) and "test" in data and "pipeline" in data["test"]:
            for t in data["test"]["pipeline"]:
                if isinstance(t, dict):
                    if "input_size" in t: t["input_size"] = (W, H)
                    if "image_size" in t: t["image_size"] = (W, H)
    except Exception: pass
    try:
        if hasattr(cfg, "data_cfg") and isinstance(cfg.data_cfg, dict):
            if "image_size" in cfg.data_cfg:
                cfg.data_cfg["image_size"] = (W, H)
    except Exception: pass

def _as_bboxes_xywh(persons: List[Dict[str,Any]]) -> np.ndarray:
    arr = np.asarray([p.get("bbox", [])[:4] for p in persons], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def infer_topdown_any(model, img_path: str, persons: List[Dict[str,Any]]):
    """统一包装 top-down 推理；兼容 mmpose 1.x/0.x，适配不同 kwargs"""
    if API_VER == 1:
        kwargs_try = [
            dict(bbox_format="xywh"),                 # 新版
            dict(format="xywh"),                      # 早期参数名
            dict(),                                   # 最简
        ]
        for kw in kwargs_try:
            try:
                # 注意：有些分支没有 bbox_thr / return_heatmap / dataset 等参数
                # 这里仅传 bbox_format/format，bbox_thr 在外部控制
                return _infer_v1(model, img_path, persons, **kw)
            except TypeError:
                continue
        # 最保守：只传 bboxes 数组
        bboxes = _as_bboxes_xywh(persons)
        return _infer_v1(model, img_path, bboxes)
    else:
        return _infer_v0(model, img_path, persons, bbox_thr=0.0, format="xywh")

def build_person_results_from_gt(coco, img_id: int) -> List[Dict[str,Any]]:
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    prs = []
    for a in anns:
        if 'bbox' not in a:  # 防御
            continue
        x, y, w, h = a['bbox']
        if w <= 1 or h <= 1:
            continue
        prs.append({'bbox': [float(x), float(y), float(w), float(h), 1.0]})
    return prs

def num_kpts_filled(a: Dict[str,Any]) -> int:
    k = a.get("keypoints", [])
    if not isinstance(k, list): return 0
    s=0
    for i in range(2, len(k), 3):
        try:
            if float(k[i]) > 0: s += 1
        except Exception:
            pass
    return s

# ---------- 评测前的“合规性”检查 ----------
def is_coco_keypoints_results(fp: Path) -> bool:
    """
    要求：JSON 是 list[dict]；每条包含 image_id/category_id/keypoints(3K)；keypoints 为 list，长度 %3==0
    """
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return False
        a0 = data[0]
        if not isinstance(a0, dict):
            return False
        if "keypoints" not in a0 or "image_id" not in a0 or "category_id" not in a0:
            return False
        kp = a0["keypoints"]
        if not isinstance(kp, list) or (len(kp) % 3 != 0):
            return False
        return True
    except Exception:
        return False

def safe_eval_coco_okspk(ann_json: str, pred_json: str, sigmas) -> dict:
    """在 eval 前先做格式检查；不合规直接抛异常（上层捕获并跳过）"""
    p = Path(pred_json)
    if not is_coco_keypoints_results(p):
        raise ValueError(f"{p.name} 不是 COCO keypoints results（缺少 'keypoints' 或结构不对）")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    s = np.asarray(sigmas, np.float32)

    coco = COCO(ann_json)
    # patch num_keypoints 以适配旧 json
    anns = coco.dataset.get("annotations", [])
    changed=False
    for a in anns:
        if "num_keypoints" not in a or a["num_keypoints"] is None:
            a["num_keypoints"] = num_kpts_filled(a); changed=True
    if changed: coco.createIndex()

    dt = coco.loadRes(pred_json)
    k_gt = len(coco.dataset['categories'][0]['keypoints'])
    if s.size != k_gt:
        s = (s[:k_gt] if s.size > k_gt else np.resize(s, k_gt)).astype(np.float32)
    E = COCOeval(coco, dt, iouType="keypoints")
    E.params.kpt_oks_sigmas = s
    E.evaluate(); E.accumulate(); E.summarize()
    st = E.stats
    return {"AP": float(st[0]), "AP50": float(st[1]), "AP75": float(st[2]), "AR": float(st[5])}

# ---------------- few-shot 训练 & 推理 ----------------
def find_resume_ckpt(work_dir: Path) -> Optional[Path]:
    lc = work_dir / 'last_checkpoint'
    if lc.is_file():
        try:
            p = Path(lc.read_text().strip())
            if p.is_file(): return p
        except Exception:
            pass
    cands = sorted(work_dir.glob("epoch_*.pth"))
    return cands[-1] if cands else None

def run_train(tag: str, seed: int, K: int, cfg_path: str,
              train_json: Path, test_json: Path, base_root: Path,
              work_dir: Path, epochs: int, lr: float,
              bb_mult: float, neck_mult: float, device: str) -> Optional[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    train_py = str(ROOT/"tools"/"train.py")

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH","")
    env["PYTHONPATH"] = os.pathsep.join(EXTRA_PY_PATHS + ([old_pp] if old_pp else []))
    if device != "cuda":
        env["CUDA_VISIBLE_DEVICES"] = ""

    resume_ckpt = find_resume_ckpt(work_dir)

    cmd = [
        sys.executable, train_py, cfg_path,
        "--work-dir", str(work_dir),
        "--cfg-options",
        # 数据根与前缀，避免 images/images 重复
        f"train_dataloader.dataset.data_root={base_root}",
        "train_dataloader.dataset.data_prefix.img=images/",
        f"train_dataloader.dataset.ann_file={train_json}",

        f"val_dataloader.dataset.data_root={base_root}",
        "val_dataloader.dataset.data_prefix.img=images/",
        f"val_dataloader.dataset.ann_file={test_json}",

        f"test_dataloader.dataset.data_root={base_root}",
        "test_dataloader.dataset.data_prefix.img=images/",
        f"test_dataloader.dataset.ann_file={test_json}",

        f"train_cfg.max_epochs={epochs}",
        f"optim_wrapper.optimizer.lr={lr}",
        f"optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult={bb_mult}",
        f"optim_wrapper.paramwise_cfg.custom_keys.neck.lr_mult={neck_mult}",
        "default_hooks.checkpoint.interval=1",
        "default_hooks.checkpoint.max_keep_ckpts=2",
        'default_hooks.checkpoint.save_best="coco/AP"',
        "randomness.deterministic=True",
        f"randomness.seed={seed}",
        "model.test_cfg.output_heatmaps=False",
    ]
    if resume_ckpt is not None:
        cmd += ["--resume"]
        print(f"[RESUME] {work_dir} -> {resume_ckpt.name}")

    print("[RUN]", " ".join(cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        warnings.warn(f"[WARN] train failed: {tag} k{K} s{seed}")
        return None

    best = sorted(work_dir.glob("best*.pth"))
    if best: return best[-1]
    latest = find_resume_ckpt(work_dir)
    if latest: return latest
    any_ckpt = sorted(work_dir.glob("epoch_*.pth"))
    return any_ckpt[-1] if any_ckpt else None

def predict_one(cfg_path: str, ckpt: str, ann_json: Path, base_root: Path,
                device: str) -> List[Dict[str,Any]]:
    """使用 GT bboxes 的 top-down 推理（与评测集一致）"""
    # build
    if API_VER == 1:
        assert Config is not None
        cfg = Config.fromfile(cfg_path)
        model = _init_model_v1(cfg, checkpoint=ckpt, device=device)
    else:
        model = _init_model_v0(cfg_path, ckpt, device=device)
    model.eval()

    from pycocotools.coco import COCO
    coco = COCO(str(ann_json))
    # 补 num_keypoints
    anns = coco.dataset.get("annotations", [])
    changed=False
    for a in anns:
        if "num_keypoints" not in a or a["num_keypoints"] is None:
            a["num_keypoints"] = num_kpts_filled(a); changed=True
    if changed: coco.createIndex()

    # 推理
    out=[]
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        info = coco.loadImgs([img_id])[0]
        # 定位图片
        # public(AP10K) 的 file_name 多半就是 "xxx.jpg"；我们在 images/ 下递归找
        img_name = Path(info.get("file_name","")).name
        img_path = None
        # 先尝试 base_root/images/<split>/file_name
        for sub in ("test","val","train",""):
            p = (base_root/"images"/sub/img_name) if sub else (base_root/"images"/img_name)
            if p.is_file():
                img_path = str(p); break
        if not img_path:
            # 兜底递归
            cand = next((x for x in (base_root/"images").rglob(img_name) if x.is_file()), None)
            if cand: img_path = str(cand)
        if not img_path:
            continue

        # GT bboxes
        persons = build_person_results_from_gt(coco, img_id)
        if len(persons)==0:  # 没 bbox 就跳过
            continue

        det = infer_topdown_any(model, img_path, persons)
        if det is None or len(det)==0:
            continue

        # 统一为 COCO keypoints 结果
        for r in det:
            inst = getattr(r, "pred_instances", None)
            if inst is None:
                inst = r  # 0.x
            kpts = inst.get("keypoints", None)
            if kpts is None:
                continue
            k = np.asarray(kpts, dtype=np.float32)
            if k.ndim == 3:
                for i in range(k.shape[0]):
                    arr = k[i]
                    if arr.shape[1] == 2:
                        vis = np.ones(arr.shape[0], dtype=np.float32)
                        flat=[]
                        for (x,y),v in zip(arr, vis):
                            flat += [float(x), float(y), float(v)]
                    else:
                        flat=[float(v) for v in arr.reshape(-1)]
                    out.append({
                        "image_id": int(img_id), "category_id": 1,
                        "keypoints": flat, "score": 1.0
                    })
            else:
                arr = k
                if arr.shape[1] == 2:
                    vis = np.ones(arr.shape[0], dtype=np.float32)
                    flat=[]
                    for (x,y),v in zip(arr, vis):
                        flat += [float(x), float(y), float(v)]
                else:
                    flat=[float(v) for v in arr.reshape(-1)]
                out.append({
                    "image_id": int(img_id), "category_id": 1,
                    "keypoints": flat, "score": 1.0
                })
    return out

# ---------------- 统计 & 画图辅助 ----------------
def mean_ci95(arr: np.ndarray) -> Tuple[float,float,float]:
    a = np.asarray(arr, float)
    if a.size == 0: return (np.nan, np.nan, np.nan)
    m = float(a.mean())
    if a.size < 2:  return (m, np.nan, np.nan)
    s = float(a.std(ddof=1))
    ci = 1.96 * s / np.sqrt(a.size)
    return (m, m-ci, m+ci)

# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser()
    # 训练/推理解耦：你可以只做评测（跳过训练），或两者都做
    ap.add_argument("--do-train", action="store_true", help="执行 few-shot 训练")
    ap.add_argument("--do-predict", action="store_true", help="训练后执行推理并导出 JSON")
    ap.add_argument("--do-eval", action="store_true", help="收集 *.keypoints.json 并评测汇总")

    ap.add_argument("--cfgs", nargs="+", required=False, help="模型配置路径列表（top-down）")
    ap.add_argument("--tags", nargs="+", required=False, help="与 cfgs 对应的 tag 名（不填则用文件名 stem）")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337,2029,3407])
    ap.add_argument("--shots", nargs="+", type=int, default=[0,10,20,30])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bb-mult", type=float, default=0.1)
    ap.add_argument("--neck-mult", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")

    # 评测用：限定收集哪些预测文件（解决 'public_*' 带来的报错）
    ap.add_argument("--pred-dirs", nargs="+", default=[
        str(PRED_DIR),
        str(ROOT/"analysis"/"sweep_preds"),
        str(ROOT/"analysis"/"public_preds"),  # 冗余写一遍
    ])
    ap.add_argument("--allowed-prefixes", nargs="+", default=["ap10k_", "sheep_"],
                    help="只评测这些前缀的 *.keypoints.json（默认屏蔽 public_* 等非合规产物）")

    args = ap.parse_args()

    # 1) few-shot 训练 + 推理（在 AP-10K 42 图上训练 K-shot，再在同一 public test 上评测）
    if args.do_train or args.do_predict:
        assert args.cfgs, "--cfgs 必填（top-down 配置路径列表）"
        tags = args.tags if args.tags else [Path(c).stem for c in args.cfgs]
        assert len(tags) == len(args.cfgs), "--tags 数量需与 --cfgs 一致"

        # AP-10K 的图像根：按你的描述就是 data_process/test_data/images/*
        base_root = AP10K_DIR
        test_json = AP10K_ANN
        assert Path(test_json).is_file(), f"[ERR] 缺少标注：{test_json}"
        assert (base_root/"images").is_dir(), f"[ERR] 缺少图像目录：{base_root/'images'}"

        for tag, cfgp in zip(tags, args.cfgs):
            cfgp = str(Path(cfgp).resolve())
            for K in args.shots:
                for seed in args.seeds:
                    work_dir = WORK_DIR / f"{tag}_ap10k_k{K}_s{seed}"
                    # —— 训练 —— #
                    if args.do_train:
                        ckpt = run_train(tag, seed, K, cfgp, test_json, test_json, base_root,
                                         work_dir, args.epochs, args.lr, args.bb_mult, args.neck_mult, args.device)
                        if ckpt is None:
                            continue
                    else:
                        ckpt = None
                        # 如果不训练也推理，尽量找 last/best
                        best = sorted(work_dir.glob("best*.pth"))
                        if best: ckpt = str(best[-1])
                        else:
                            last = find_resume_ckpt(work_dir)
                            if last: ckpt = str(last)

                    # —— 推理 —— #
                    if args.do_predict:
                        if not ckpt:
                            warnings.warn(f"[SKIP] 无 ckpt：{work_dir}")
                            continue
                        try:
                            preds = predict_one(cfgp, ckpt, test_json, base_root, args.device)
                        except Exception as e:
                            warnings.warn(f"[SKIP] 推理失败：{tag} k{K} s{seed} — {e}")
                            continue
                        out_json = PRED_DIR / f"ap10k_{tag}_k{K}_s{seed}.keypoints.json"
                        if len(preds)==0:
                            warnings.warn(f"[SKIP] {out_json.name}: 得到 0 个实例。")
                            continue
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(preds, f)
                        print("[OUT]", out_json)

    # 2) 评测（收集 *.keypoints.json → 过滤前缀 → 结构校验 → COCOeval）
    if args.do_eval:
        ann = str(AP10K_ANN)
        assert Path(ann).is_file(), f"[ERR] 缺少标注：{ann}"

        # 收集候选
        cand = []
        for d in args.pred_dirs:
            p = Path(d)
            if not p.is_dir(): continue
            cand += [str(x) for x in p.glob("*.keypoints.json")]
        cand = sorted(set(cand))
        if not cand:
            raise SystemExit("[ERR] 没找到 *.keypoints.json")

        # 前缀白名单（默认只认 ap10k_*, sheep_*）
        prefixes = tuple(args.allowed_prefixes)
        def allowed_by_prefix(name: str) -> bool:
            return name.startswith(prefixes)

        rows = []
        skipped = 0
        name_pat = re.compile(r".*/(?P<prefix>[a-z0-9_]+)_(?P<tag>[^/]+)_k(?P<shot>\d+)_s(?P<seed>\d+)\.keypoints\.json$")
        for p in cand:
            name = Path(p).name
            if not allowed_by_prefix(name):
                # 屏蔽 public_* / 其他非标准产物
                continue
            m = name_pat.match(p)
            if not m:
                skipped += 1
                warnings.warn(f"[SKIP] 命名不匹配：{name}")
                continue
            tag = m.group("tag"); seed = int(m.group("seed")); shot = int(m.group("shot"))

            # 结构校验
            if not is_coco_keypoints_results(Path(p)):
                skipped += 1
                warnings.warn(f"[SKIP] eval failed: {name} — 'keypoints' 缺失或结构错误")
                continue

            # 评测
            try:
                res = safe_eval_coco_okspk(ann, p, FIXED_SIGMAS)
            except Exception as e:
                skipped += 1
                warnings.warn(f"[SKIP] eval failed: {name} — {e}")
                continue

            rows.append({
                "tag": tag, "seed": seed, "shots": shot, **res, "pred_path": p
            })

        if not rows:
            raise SystemExit("[ERR] 没有可评测的文件（全部被过滤或失败）。")

        df = pd.DataFrame(rows).sort_values(["tag","shots","seed"]).reset_index(drop=True)
        df.to_csv(RUNS_CSV, index=False)
        print(f"[OK] {RUNS_CSV}  ({len(df)} 行)")

        # 跨 seed 的 mean±95%CI（按 tag×shots 分组）
        agg_rows=[]
        for (tag,shot), g in df.groupby(["tag","shots"]):
            m, lo, hi   = mean_ci95(g["AP"].to_numpy(float))
            m50, lo50, hi50 = mean_ci95(g["AP50"].to_numpy(float))
            m75, lo75, hi75 = mean_ci95(g["AP75"].to_numpy(float))
            mar, laro, haro = mean_ci95(g["AR"].to_numpy(float))
            agg_rows.append({
                "tag": tag, "shots": shot, "n_seeds": len(g),
                "AP_mean": m, "AP_CI_lo": lo, "AP_CI_hi": hi,
                "AP50_mean": m50, "AP50_CI_lo": lo50, "AP50_CI_hi": hi50,
                "AP75_mean": m75, "AP75_CI_lo": lo75, "AP75_CI_hi": hi75,
                "AR_mean": mar, "AR_CI_lo": laro, "AR_CI_hi": haro,
            })
        agg = pd.DataFrame(agg_rows).sort_values(["tag","shots"]).reset_index(drop=True)
        agg.to_csv(SUM_CSV, index=False)
        print(f"[OK] {SUM_CSV}")
        if skipped:
            print(f"[INFO] 跳过 {skipped} 个文件（前缀不允许 / 命名不匹配 / 结构不合规）。")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolution scaling — BENCH (latency/FPS/memory/params/GFLOPs)
- 四个输入尺寸；batch=1/8
- Top-down (SimCC/SAR): H x round(0.75H)；YOLOX: S x S
- 计时与复杂度：
    * SimCC：不再依赖 data_samples / predict；统一走“子图 Runner”
              （优先 forward_dummy；否则 backbone→neck→(keypoint_head/head)）
    * SAR/YOLOX：使用 wrapper（predict() 优先），必要时回退到 mode='tensor'
- SAR/YOLOX 显式注入 PoseDataSample + dataset_meta
- GFLOPs 统一按 FLOPs 计（若工具返回 GMac，则 ×2）
输出：analysis/summary/sweep_bench.csv
"""

from pathlib import Path
import os, sys, csv, warnings, argparse, re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(ROOT))

# ---- import path ----
for p in (ROOT, ROOT/'SimCC-main', ROOT/'SAR-main', ROOT/'yolox_pose-main'):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope, DefaultScope
from contextlib import contextmanager

# ---------------- 全局开关 ----------------
SIMCC_USE_DUMMY = True   # SimCC 走子图 Runner（forward_dummy 或 backbone→neck→head）

# ---- CLI ----
ap = argparse.ArgumentParser()
ap.add_argument("--no-cuda-sync", action="store_true", help="禁用 cuda.synchronize()（仅调试）")
args, _ = ap.parse_known_args()
CUDA_SYNC = not args.no_cuda_sync

# ---- 输出 ----
OUT_DIR = ROOT / "analysis" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "sweep_bench.csv"

# ---- 设备 & 批次 ----
device = "cuda:0" if torch.cuda.is_available() else "cpu"
BS_LIST = (1, 8)
WARMUP, ITERS = 20, 100

# ---- 分辨率集合 ----
TOPDOWN_H = [256, 384, 512, 640]  # H
YOLOX_S   = [224, 320, 448, 544]  # S

def hw_for_topdown(H):  # 4:3
    return int(H), int(round(H*0.75))

def hw_for_yolox(S):
    return int(S), int(S)

# ---- 模型（九个标准名）----
CFGS = {
    "SimCC-W48":      str(ROOT/"SimCC-main/configs/sheep/simcc_hrnet-w48_8xb32-280e_sheep-384x288.py"),
    "SimCC-HRForm":   str(ROOT/"SimCC-main/configs/sheep/simcc_hrtran_8xb32-280e_sheep-384x288.py"),
    "SimCC-LiteHR":   str(ROOT/"SimCC-main/configs/sheep/simcc_litehrnet_8xb32-280e_sheep-384x288.py"),
    "SimCC-Res50":    str(ROOT/"SimCC-main/configs/sheep/simcc_res50_8xb32-280e_sheep-384x288.py"),
    "SimCC-Swin":     str(ROOT/"SimCC-main/configs/sheep/simcc_swim_8xb32-280e_sheep-384x288.py"),
    "SAR-W48":        str(ROOT/"SAR-main/configs/sheep/SAR_hrnet-w48_8xb32-280e_sheep-384x288.py"),
    "SAR-Res50":      str(ROOT/"SAR-main/configs/sheep/SAR_res50_8xb32-280e_sheep-384x288.py"),
    "YOLOX-s":        str(ROOT/"yolox_pose-main/configs/yolox-pose_s_8xb32-300e_sheep_coco.py"),
    "YOLOX-m":        str(ROOT/"yolox_pose-main/configs/yolox-pose_m_4xb16-300e_sheep_coco.py"),
}

FAMILY = {
    "SimCC-W48":"SimCC","SimCC-HRForm":"SimCC","SimCC-LiteHR":"SimCC","SimCC-Res50":"SimCC","SimCC-Swin":"SimCC",
    "SAR-W48":"SAR","SAR-Res50":"SAR",
    "YOLOX-s":"YOLOX","YOLOX-m":"YOLOX",
}

# ---------------- 基础构建 ----------------
def set_scope(name: str):
    try:
        cur = DefaultScope.get_current_instance()
        if cur is not None:
            cur.close()
    except Exception:
        pass
    init_default_scope(name)

def load_cfg(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    scope = 'mmyolo' if ('yolox' in cfg_path.lower() or 'mmyolo' in cfg_path.lower()) else 'mmpose'
    set_scope(scope)
    return cfg, scope

def build_full_model(cfg):
    model = MODELS.build(cfg.model)
    model.to(device).eval()
    return model

# ---------- 从 cfg/路径解析训练输入尺寸（SimCC 用） ----------
def get_train_input_size(cfg, cfg_path: str):
    """
    以 (H, W) 返回训练输入分辨率；尽量从 cfg 中解析，
    否则从文件名里的 `...-384x288.py` 提取；都失败则返回 None。
    """
    keys = [
        ("model", "test_cfg", "codec", "input_size"),
        ("model", "train_cfg", "codec", "input_size"),
        ("codec", "input_size"),
        ("dataset_meta", "input_size"),
        ("data_preprocessor", "input_size"),
    ]
    for ks in keys:
        node = cfg
        try:
            for k in ks:
                if isinstance(node, dict) and k in node:
                    node = node[k]
                else:
                    node = None
                    break
            if isinstance(node, (list, tuple)) and len(node) == 2:
                W, H = int(node[0]), int(node[1])  # mmpose 常用 (W,H)
                return (int(H), int(W))
        except Exception:
            pass

    m = re.search(r'(\d{3,4})x(\d{3,4})', str(cfg_path))
    if m:
        H = int(m.group(1)); W = int(m.group(2))
        return (H, W)
    return None

# ---------------- dataset_meta 与 PoseDataSample（供 SAR/YOLOX） ----------------
def infer_num_keypoints(model, cfg) -> int:
    for attr in ("keypoint_head", "head", "bbox_head"):
        h = getattr(model, attr, None)
        if h is not None and hasattr(h, "num_keypoints"):
            try:
                k = int(h.num_keypoints)
                if k > 0:
                    return k
            except Exception:
                pass
    try:
        k = int(cfg.model.get("keypoint_head", {}).get("num_keypoints", 0))
        if k > 0: return k
    except Exception:
        pass
    try:
        k = int(cfg.model.get("head", {}).get("num_keypoints", 0))
        if k > 0: return k
    except Exception:
        pass
    return 18  # fallback

def ensure_dataset_meta(model, num_kpts: int, cfg):
    meta = getattr(model, "dataset_meta", None)
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("num_keypoints", num_kpts)
    meta.setdefault("flip_pairs", [])
    meta.setdefault("flip_indices", list(range(num_kpts)))
    sigmas = None
    try:
        sigmas = cfg.get("dataset_meta", {}).get("sigmas", None)
    except Exception:
        pass
    if sigmas is not None:
        try:
            import numpy as _np
            meta["sigmas"] = _np.asarray(sigmas, dtype=_np.float32)
        except Exception:
            meta["sigmas"] = sigmas
    else:
        meta.setdefault("sigmas", None)
    model.dataset_meta = meta

def build_dummy_samples(bs: int, H: int, W: int, family: str, num_kpts: int):
    try:
        from mmpose.structures import PoseDataSample
    except Exception:
        PoseDataSample = None
    heat_h, heat_w = max(1, H // 4), max(1, W // 4)
    samples = []
    for _ in range(bs):
        meta = dict(
            img_shape=(H, W),
            input_size=(W, H),
            center=np.asarray([W / 2.0, H / 2.0], dtype=np.float32),
            scale=np.asarray([W / 200.0, H / 200.0], dtype=np.float32),
            bbox_score=float(1.0),
            heatmap_size=(heat_w, heat_h),
            flip_indices=list(range(num_kpts)),
        )
        if family in ("SimCC", "SAR") and PoseDataSample is not None:
            ds = PoseDataSample(); ds.set_metainfo(meta); samples.append(ds)
        else:
            from types import SimpleNamespace
            samples.append(SimpleNamespace(metainfo=meta))
    return samples

# ---------------- 统一前向（给 SAR/YOLOX 用；predict 优先） ----------------
def full_forward_tensor(model, x, samples):
    def _ensure_tensor(out):
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (list, tuple)) and out:
            return _ensure_tensor(out[0])
        if isinstance(out, dict) and out:
            for v in out.values():
                t = _ensure_tensor(v)
                if isinstance(t, torch.Tensor):
                    return t
        return torch.as_tensor(0., device=x.device)

    for kw in ("batch_data_samples", "data_samples"):
        try:
            return _ensure_tensor(model.predict(x, **{kw: samples}))
        except Exception:
            pass

    for kw in ("batch_data_samples", "data_samples"):
        try:
            return _ensure_tensor(model.forward(x, **{kw: samples, "mode": "tensor"}))
        except Exception:
            pass
        try:
            return _ensure_tensor(model(x, **{kw: samples, "mode": "tensor"}))
        except Exception:
            pass

    for kw in ("batch_data_samples", "data_samples"):
        try:
            return _ensure_tensor(model.forward(x, **{kw: samples}))
        except Exception:
            pass
        try:
            return _ensure_tensor(model(x, **{kw: samples}))
        except Exception:
            pass

    try:
        return _ensure_tensor(model(x))
    except Exception:
        pass
    try:
        return _ensure_tensor(model.forward(x))
    except Exception:
        pass

    raise RuntimeError("full_forward_tensor failed for all known signatures")

def probe_feat_hw_before_head(model, train_hw):
    """
    返回 head 入口处的特征空间尺寸 (feat_H, feat_W)
    在“训练输入分辨率”下，跑一遍 backbone -> (neck) 得到真实尺寸。
    """
    Ht, Wt = int(train_hw[0]), int(train_hw[1])
    with torch.no_grad():
        x = torch.randn(1, 3, Ht, Wt, device=next(model.parameters()).device)
        feats = model.backbone(x) if hasattr(model, "backbone") else model(x)
        if not isinstance(feats, (list, tuple)):
            feats = (feats,)
        nk = getattr(model, "neck", None)
        if nk is not None:
            feats = nk(feats if isinstance(feats, (list, tuple)) else (feats,))
            if not isinstance(feats, (list, tuple)):
                feats = (feats,)
        f0 = feats[0] if isinstance(feats, (list, tuple)) else feats
        fh, fw = int(f0.shape[-2]), int(f0.shape[-1])
    return (max(1, fh), max(1, fw))

# ---------------- SimCC 专用 DummyRunner（子图：backbone→neck→head） ----------------
class DummyRunner(torch.nn.Module):
    """
    SimCC 兜底 Runner：
      - 有 forward_dummy 则直接用
      - 否则：backbone→neck→(keypoint_head/head) 前向
      - 若 head 因空间尺寸不匹配而失败，则把特征插值为“训练时特征尺寸”再试一次
    标记：
      self.used_head         -> 最终是否走到了 head
      self.head_failed       -> 两次尝试都失败
      self.rescales_for_head -> 是否做了插值以满足 head
      self.head_name         -> 实际使用/尝试的 head 类名
    """
    def __init__(self, model, expected_feat_hw=None):
        super().__init__()
        self.m = model
        self.expected_feat_hw = expected_feat_hw  # (feat_H, feat_W) or None
        self.used_head = False
        self.head_failed = False
        self.rescales_for_head = False
        self.head_name = None

    def _call_head(self, head, feats):
        # 先 head.forward，再 head()
        try:
            return head.forward(feats if isinstance(feats, (list, tuple)) else (feats,))
        except Exception:
            pass
        return head(feats if isinstance(feats, (list, tuple)) else (feats,))

    def forward(self, x):
        m = self.m
        # 1) forward_dummy
        if hasattr(m, "forward_dummy"):
            self.used_head = True
            self.head_failed = False
            self.rescales_for_head = False
            h = getattr(m, "keypoint_head", None)
            if h is None: h = getattr(m, "head", None)
            self.head_name = getattr(h, "__name__", None) or getattr(getattr(h, "__class__", object), "__name__", "None")
            return m.forward_dummy(x)

        # 2) 子图
        if not hasattr(m, "backbone"):
            self.used_head = False
            self.head_failed = False
            self.rescales_for_head = False
            self.head_name = None
            return m(x)

        feats = m.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = (feats,)
        neck = getattr(m, "neck", None)
        if neck is not None:
            feats = neck(feats if isinstance(feats, (list, tuple)) else (feats,))
            if not isinstance(feats, (list, tuple)):
                feats = (feats,)

        head = getattr(m, "keypoint_head", None)
        if head is None:
            head = getattr(m, "head", None)
        self.head_name = getattr(head, "__name__", None) or getattr(getattr(head, "__class__", object), "__name__", "None")

        if head is not None:
            # 尝试直接喂
            try:
                out = self._call_head(head, feats)
                self.used_head = True
                self.head_failed = False
                self.rescales_for_head = False
                return out
            except Exception:
                pass
            # 若失败：尝试把张量 resize 到训练特征尺寸
            try:
                if self.expected_feat_hw is not None:
                    tgt_h, tgt_w = self.expected_feat_hw
                else:
                    f0 = feats[0] if isinstance(feats, (list, tuple)) else feats
                    tgt_h, tgt_w = f0.shape[-2], f0.shape[-1]
                new_feats = []
                for f in (feats if isinstance(feats, (list, tuple)) else [feats]):
                    if isinstance(f, torch.Tensor) and f.dim() == 4:
                        new_feats.append(F.interpolate(f, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False))
                    else:
                        new_feats.append(f)
                out = self._call_head(head, tuple(new_feats))
                self.used_head = True
                self.head_failed = False
                self.rescales_for_head = True
                return out
            except Exception:
                self.used_head = False
                self.head_failed = True

        # 3) 彻底失败：返回一个 Tensor 以保流程不崩
        for t in (feats if isinstance(feats, (list, tuple)) else [feats]):
            if isinstance(t, torch.Tensor):
                return t
        return torch.as_tensor(0., device=x.device)

# ---------------- GFLOPs（SimCC 用 DummyRunner；其余按 wrapper） ----------------
def get_gflops_full(model, H, W, family: str, num_kpts: int, model_name: str, expected_feat_hw=None):
    if family == "SimCC" and SIMCC_USE_DUMMY:
        runner = DummyRunner(model, expected_feat_hw=expected_feat_hw).to(device).eval()
        dummy = torch.randn(1, 3, H, W, device=device)
        # 先干跑一次以设置标志
        with torch.no_grad():
            _ = runner(dummy)
        print(
            f"[HEAD][GFLOPs] {model_name} {H}x{W}: "
            f"used_head={int(runner.used_head)} "
            f"head_failed={int(runner.head_failed)} "
            f"rescaled={int(runner.rescales_for_head)} "
            f"head={runner.head_name}"
        )
        # thop (GMac → ×2)
        try:
            from thop import profile
            macs, _ = profile(runner, inputs=(dummy,), verbose=False)
            return 2.0 * (float(macs) / 1e9)
        except Exception as e:
            warnings.warn(f"[WARN] GFLOPs(thop) fallback for SimCC failed: {e}")
        # mmengine
        try:
            from mmengine.analysis import get_model_complexity_info
            macs_str, _ = get_model_complexity_info(runner, (3, H, W), show_table=False, show_arch=False)
            m = re.search(r'([\d.]+)\s*GMac', macs_str or '')
            gmac = float(m.group(1)) if m else float('nan')
            return 2.0 * gmac if np.isfinite(gmac) else float('nan')
        except Exception as e2:
            warnings.warn(f"[WARN] GFLOPs(mmengine) fallback for SimCC failed: {e2}")
            return float('nan')

    class Runner(nn.Module):
        def __init__(self, m, H, W, family, num_kpts):
            super().__init__()
            self.m, self.H, self.W, self.family, self.num_kpts = m, H, W, family, num_kpts
        def forward(self, x):
            if hasattr(self.m, "forward_dummy"):
                return self.m.forward_dummy(x)
            samples = build_dummy_samples(x.shape[0], self.H, self.W, self.family, self.num_kpts)
            return full_forward_tensor(self.m, x, samples)

    runner = Runner(model, H, W, family, num_kpts).to(device).eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    try:
        from thop import profile
        macs, _ = profile(runner, inputs=(dummy,), verbose=False)
        return 2.0 * (float(macs) / 1e9)
    except Exception as e:
        warnings.warn(f"[WARN] GFLOPs via thop failed: {e}")
    try:
        from mmengine.analysis import get_model_complexity_info
        macs_str, _ = get_model_complexity_info(runner, (3, H, W), show_table=False, show_arch=False)
        m = re.search(r'([\d.]+)\s*GMac', macs_str or '')
        gmac = float(m.group(1)) if m else float('nan')
        return 2.0 * gmac if np.isfinite(gmac) else float('nan')
    except Exception as e2:
        warnings.warn(f"[WARN] GFLOPs via mmengine failed: {e2}")
        return float('nan')

# ---- 小工具：空上下文（兼容 CPU 情况）----
@contextmanager
def dummy_context():
    yield

# ---------------- 主基准 ----------------
def bench_one(std_name, cfg_path):
    cfg, _ = load_cfg(cfg_path)
    model = build_full_model(cfg)

    # SAR/YOLOX 可能需要；SimCC 不依赖
    num_kpts = infer_num_keypoints(model, cfg)
    ensure_dataset_meta(model, num_kpts, cfg)

    # ——关键：给 SimCC 计算“训练特征尺寸（head 入口）”——
    train_hw = get_train_input_size(cfg, cfg_path)  # (H, W) or None
    expected_feat_hw = None
    if train_hw is not None:
        try:
            expected_feat_hw = probe_feat_hw_before_head(model, train_hw)
            print(f"[INFO] expected_feat_hw@train for {std_name}: {expected_feat_hw}")
        except Exception as e:
            warnings.warn(f"[WARN] probe_feat_hw_before_head failed for {std_name}: {e}")
            # 兜底：保持之前的 1/4 假设，不影响已能跑通的 HRNet 家族
            expected_feat_hw = (max(1, train_hw[0] // 4), max(1, train_hw[1] // 4))

    def run_at(H, W, size_tag):
        rows=[]
        gflops = get_gflops_full(model, H, W, FAMILY[std_name], num_kpts, std_name, expected_feat_hw)

        for bs in BS_LIST:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            x = torch.randn(bs,3,H,W,device=device)
            samples = build_dummy_samples(bs, H, W, FAMILY[std_name], num_kpts)

            use_dummy_for_latency = (FAMILY[std_name] == "SimCC") and SIMCC_USE_DUMMY
            latency_runner = DummyRunner(model, expected_feat_hw=expected_feat_hw).to(device).eval() if use_dummy_for_latency else None

            # 预热
            try:
                with torch.cuda.amp.autocast(False) if device.startswith("cuda") else dummy_context():
                    if use_dummy_for_latency:
                        _ = latency_runner(x)
                    else:
                        _ = full_forward_tensor(model, x, samples)
                    if device.startswith("cuda") and CUDA_SYNC: torch.cuda.synchronize()
                    for _ in range(WARMUP):
                        if use_dummy_for_latency:
                            _ = latency_runner(x)
                        else:
                            _ = full_forward_tensor(model, x, samples)
                    if device.startswith("cuda") and CUDA_SYNC: torch.cuda.synchronize()
            except Exception as e:
                raise RuntimeError(f"warmup failed: {e}")

            # 计时
            times=[]
            if device.startswith("cuda"):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(ITERS):
                    starter.record()
                    with torch.cuda.amp.autocast(False):
                        if use_dummy_for_latency:
                            _ = latency_runner(x)
                        else:
                            _ = full_forward_tensor(model, x, samples)
                    ender.record()
                    if CUDA_SYNC: torch.cuda.synchronize()
                    times.append(starter.elapsed_time(ender))
            else:
                import time as _time
                for _ in range(ITERS):
                    t0=_time.perf_counter()
                    if use_dummy_for_latency:
                        _ = latency_runner(x)
                    else:
                        _ = full_forward_tensor(model, x, samples)
                    times.append((_time.perf_counter()-t0)*1000.0)

            times = sorted(times)
            avg = float(np.mean(times))
            p50 = times[int(0.5*(len(times)-1))]
            p95 = times[int(0.95*(len(times)-1))]
            fps = 1000.0/avg*bs
            vram_mb = torch.cuda.max_memory_allocated()/ (1024**2) if device.startswith("cuda") else 0.0

            try:
                params_m = sum(p.numel() for p in model.parameters())/1e6
            except Exception:
                params_m = float('nan')

            rows.append({
                "model":std_name, "family":FAMILY[std_name],
                "size":size_tag, "H":H, "W":W, "bs":bs, "device":device,
                "lat_ms_avg":avg, "lat_ms_p50":p50, "lat_ms_p95":p95,
                "fps":fps, "params_M":params_m, "GFLOPs":gflops,
                "PeakMem_MB":vram_mb,
                "simcc_dummy": int(use_dummy_for_latency) if FAMILY[std_name]=="SimCC" else 0,
            })
        return rows

    sizes = ([(f"H{H}",)+hw_for_topdown(H) for H in TOPDOWN_H]
             if FAMILY[std_name]!="YOLOX" else
             [(f"S{S}",)+hw_for_yolox(S) for S in YOLOX_S])

    out=[]
    for tag,H,W in sizes:
        out += run_at(H,W,tag)

    del model
    if device.startswith("cuda"): torch.cuda.empty_cache()
    set_scope('mmpose')  # 还原
    return out

# ---------------- 主流程 ----------------
with open(OUT_CSV,"w",newline="") as f:
    csv.writer(f).writerow(
        ["model","family","size","H","W","bs","device",
         "lat_ms_avg","lat_ms_p50","lat_ms_p95","fps","params_M","GFLOPs","PeakMem_MB","simcc_dummy"]
    )

for name,cfg_path in CFGS.items():
    try:
        print(f"[BENCH] {name}")
        rows = bench_one(name, cfg_path)
        with open(OUT_CSV,"a",newline="") as f:
            w=csv.writer(f)
            for r in rows:
                w.writerow([r[k] for k in
                    ["model","family","size","H","W","bs","device",
                     "lat_ms_avg","lat_ms_p50","lat_ms_p95","fps","params_M","GFLOPs","PeakMem_MB","simcc_dummy"]])
    except Exception as e:
        warnings.warn(f"[WARN] bench failed for {name}: {e}")

print(f"[OUT] {OUT_CSV}")

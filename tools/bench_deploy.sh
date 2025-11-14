#!/usr/bin/env bash
set -eo pipefail

# ---------- ROOT ----------
if [[ -n "${BASH_SOURCE:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "[ROOT] ${ROOT}"
cd "${ROOT}"

# 关键：补全 PYTHONPATH，保证 SAR / YOLOX 自定义模块可被 import
export PYTHONPATH="${ROOT}:${ROOT}/SimCC-main:${ROOT}/SAR-main:${ROOT}/yolox_pose-main:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="${ROOT}/analysis/deploy"
OUT_CSV="${OUT_DIR}/bench_results.csv"
mkdir -p "${OUT_DIR}"

# ---------- 统一九个模型（sheep 配置） ----------
SIMCC_DIR="${ROOT}/SimCC-main/configs/sheep"
SAR_DIR="${ROOT}/SAR-main/configs/sheep"
YOLOX_DIR="${ROOT}/yolox_pose-main/configs"

declare -A CFGS=(
  ["SimCC-HRNet-W48"]="${SIMCC_DIR}/simcc_hrnet-w48_8xb32-280e_sheep-384x288.py"
  ["SimCC-HRTrans"]="${SIMCC_DIR}/simcc_hrtran_8xb32-280e_sheep-384x288.py"
  ["SimCC-LiteHRNet"]="${SIMCC_DIR}/simcc_litehrnet_8xb32-280e_sheep-384x288.py"
  ["SimCC-ResNet-50"]="${SIMCC_DIR}/simcc_res50_8xb32-280e_sheep-384x288.py"
  ["SimCC-Swin"]="${SIMCC_DIR}/simcc_swim_8xb32-280e_sheep-384x288.py"
  ["SAR-HRNet-W48"]="${SAR_DIR}/SAR_hrnet-w48_8xb32-280e_sheep-384x288.py"
  ["SAR-ResNet-50"]="${SAR_DIR}/SAR_res50_8xb32-280e_sheep-384x288.py"
  ["YOLOX-Pose-s"]="${YOLOX_DIR}/yolox-pose_s_8xb32-300e_sheep_coco.py"
  ["YOLOX-Pose-m"]="${YOLOX_DIR}/yolox-pose_m_4xb16-300e_sheep_coco.py"
)

python - <<'PY'
import os, sys, time, csv, math, warnings, re, gc
from pathlib import Path

# --------- sys.path 保险插入（与上面的 PYTHONPATH 相同）---------
ROOT = Path(os.getcwd())
for p in (ROOT, ROOT/'SimCC-main', ROOT/'SAR-main', ROOT/'yolox_pose-main'):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# --------- 进程内存（RAM）监控：优先 psutil，回退 resource ----------
def _make_rss_reader():
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        def rss_mb():
            return proc.memory_info().rss / (1024**2)
        return rss_mb
    except Exception:
        try:
            import resource
            def rss_mb():
                # ru_maxrss: Linux 下为 KB；此处转换为 MB
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            return rss_mb
        except Exception:
            def rss_mb():
                return float('nan')
            return rss_mb

rss_mb = _make_rss_reader()

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS as MM_MODELS, DefaultScope
from mmengine.utils import import_modules_from_strings

OUT_DIR = ROOT / "analysis" / "deploy"
OUT_CSV = OUT_DIR / "bench_results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFGS = {
    "SimCC-HRNet-W48": str(ROOT / "SimCC-main/configs/sheep/simcc_hrnet-w48_8xb32-280e_sheep-384x288.py"),
    "SimCC-HRTrans":   str(ROOT / "SimCC-main/configs/sheep/simcc_hrtran_8xb32-280e_sheep-384x288.py"),
    "SimCC-LiteHRNet": str(ROOT / "SimCC-main/configs/sheep/simcc_litehrnet_8xb32-280e_sheep-384x288.py"),
    "SimCC-ResNet-50": str(ROOT / "SimCC-main/configs/sheep/simcc_res50_8xb32-280e_sheep-384x288.py"),
    "SimCC-Swin":      str(ROOT / "SimCC-main/configs/sheep/simcc_swim_8xb32-280e_sheep-384x288.py"),
    "SAR-HRNet-W48":   str(ROOT / "SAR-main/configs/sheep/SAR_hrnet-w48_8xb32-280e_sheep-384x288.py"),
    "SAR-ResNet-50":   str(ROOT / "SAR-main/configs/sheep/SAR_res50_8xb32-280e_sheep-384x288.py"),
    "YOLOX-Pose-s":    str(ROOT / "yolox_pose-main/configs/yolox-pose_s_8xb32-300e_sheep_coco.py"),
    "YOLOX-Pose-m":    str(ROOT / "yolox_pose-main/configs/yolox-pose_m_4xb16-300e_sheep_coco.py"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

def set_scope(scope: str):
    try:
        cur = DefaultScope.get_current_instance()
        if cur is not None:
            cur.close()
    except Exception:
        pass
    init_default_scope(scope)

def need_mmyolo(cfg_path: str, cfg: "Config") -> bool:
    p = str(cfg_path).lower()
    if "yolox" in p or "mmyolo" in p:
        return True
    try:
        t = (cfg.model or {}).get('type', '')
        if isinstance(t, str) and t.lower().startswith('yolo'):
            return True
    except Exception:
        pass
    return False

def prepare_custom_imports(cfg: "Config"):
    ci = getattr(cfg, 'custom_imports', None)
    if isinstance(ci, dict):
        mods = ci.get('imports', [])
        if mods:
            import_modules_from_strings(mods, allow_failed_imports=False)

def safe_build_model(name: str, cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    prepare_custom_imports(cfg)

    if need_mmyolo(cfg_path, cfg):
        try:
            __import__('mmyolo')
        except Exception as e:
            raise RuntimeError(
                "需要 mmyolo 才能构建 YOLOX 模型；请先安装：pip install -U mmyolo"
            ) from e
        set_scope('mmyolo')
    else:
        set_scope('mmpose')

    try:
        model = MM_MODELS.build(cfg.model)
    except Exception as e:
        try:
            set_scope('mmdet')
            model = MM_MODELS.build(cfg.model)
        except Exception:
            raise e
    model.to(device).eval()
    return model, cfg

def get_input_size_from_cfg(cfg: Config):
    W = H = None
    try:
        if "codec" in cfg and "input_size" in cfg.codec:
            w, h = cfg.codec.input_size
            W, H = int(w), int(h)
    except Exception: pass
    if W is None or H is None:
        try:
            if hasattr(cfg, "img_scale") and cfg.img_scale is not None:
                W, H = int(cfg.img_scale[0]), int(cfg.img_scale[1])
        except Exception: pass
    if W is None or H is None:
        try:
            text = Path(cfg.filename).read_text()
            m = re.search(r'input_size=\[?(\d+)\s*,\s*(\d+)\]?', text)
            if m: W, H = int(m.group(1)), int(m.group(2))
            else:
                m = re.search(r'img_scale=\[?(\d+)\s*,\s*(\d+)\]?', text)
                if m: W, H = int(m.group(1)), int(m.group(2))
        except Exception: pass
    if W is None or H is None:
        W, H = 288, 384
    return (H, W)

def forward_once(model, x):
    with torch.no_grad():
        try: return model(x, mode='tensor')
        except Exception: pass
        if hasattr(model, "forward_dummy"):
            try: return model.forward_dummy(x)
            except Exception: pass
        feat = None
        try: feat = model.backbone(x)
        except Exception: pass
        if feat is not None:
            if getattr(model, "neck", None) is not None:
                try: feat = model.neck(feat)
                except Exception: pass
            if getattr(model, "head", None) is not None:
                try:
                    return model.head(feat if isinstance(feat, (list,tuple)) else [feat])
                except Exception: pass
        return model(x)

def bench_one(model, cfg, name: str, hw, batch_size: int, warmup=20, iters=100):
    H, W = hw

    # ---- CPU 内存“峰值增量 ΔRSS”基线：在分配输入之前取 ----
    gc.collect()
    baseline_rss = rss_mb()     # 基线（MB）
    peak_rss     = baseline_rss # 峰值初始化

    # ---- GPU：清零显存统计；CPU：不用动 ----
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- 分配输入张量 & 预热 ----
    x = torch.randn(batch_size, 3, H, W, device=device)
    _ = forward_once(model, x)
    if device == "cuda":
        torch.cuda.synchronize()
    else:
        peak_rss = max(peak_rss, rss_mb())

    for _ in range(warmup):
        _ = forward_once(model, x)
        if device == "cuda":
            torch.cuda.synchronize()
        else:
            peak_rss = max(peak_rss, rss_mb())

    # ---- 正式计时 ----
    times = []
    if device == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        if device == "cuda":
            starter.record()
            _ = forward_once(model, x)
            ender.record(); torch.cuda.synchronize()
            dt_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            _ = forward_once(model, x)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            peak_rss = max(peak_rss, rss_mb())
        times.append(dt_ms)

    # ---- 汇总：时延/FPS ----
    avg_ms = float(sum(times)/len(times))
    fps    = float(1000.0 / avg_ms * batch_size)

    # ---- 统一的“峰值内存”字段：
    # GPU => 显存峰值（MB）；CPU => ΔRSS = 峰值RSS - 基线RSS（MB）
    if device == "cuda":
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        host_ram_delta_mb = 0.0   # GPU 情况下不报告 RAM 增量
    else:
        vram_peak_mb = 0.0
        host_ram_delta_mb = max(0.0, peak_rss - baseline_rss)

    # 参数量（M）
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "method": name,
        "input_hw": f"{H}x{W}",
        "batch": batch_size,
        "latency_ms": round(avg_ms, 3),
        "fps": round(fps, 1),
        "max_mem_MB": round(vram_peak_mb, 1),     # 显存（GPU）
        "host_ram_MB": round(host_ram_delta_mb, 1),# ΔRSS（CPU）
        "params_M": round(params_m, 2),
        "device": device,
    }

# 写 CSV 表头（host_ram_MB = CPU ΔRSS；GPU 为 0）
if not OUT_CSV.exists():
    with OUT_CSV.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["method","input_hw","batch","latency_ms","fps","max_mem_MB","host_ram_MB","params_M","device"]
        )

for name, cfg_path in CFGS.items():
    if not Path(cfg_path).is_file():
        print(f"[SKIP] cfg missing: {name} -> {cfg_path}")
        continue
    try:
        model, cfg = safe_build_model(name, cfg_path)
        H, W = get_input_size_from_cfg(cfg)
        print(f"[BENCH] {name} | {H}x{W}")
        rows = []
        for bs in (1, 8):
            try:
                row = bench_one(model, cfg, name, (H,W), bs)
                rows.append(row)
                print(f"  - bs={bs}: {row['latency_ms']} ms | {row['fps']} FPS | "
                      f"VRAM={row['max_mem_MB']} MB | RAMΔ={row['host_ram_MB']} MB")
            except Exception as e:
                warnings.warn(f"[WARN] bench failed: {name} bs={bs}: {e}")
        if rows:
            with OUT_CSV.open("a", newline="") as f:
                w = csv.writer(f)
                for r in rows:
                    w.writerow([
                        r["method"], r["input_hw"], r["batch"], r["latency_ms"], r["fps"],
                        r["max_mem_MB"], r["host_ram_MB"], r["params_M"], r["device"]
                    ])
        del model
        if device == "cuda": torch.cuda.empty_cache()
        # bench 完后把域切回 mmpose，以免影响下一个
        set_scope('mmpose')
    except Exception as e:
        warnings.warn(f"[WARN] model build or bench crashed: {name}: {e}")

print(f"[OUT] {OUT_CSV}")
PY

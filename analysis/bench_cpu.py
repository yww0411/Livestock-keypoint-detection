#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU-only benchmark: latency / FPS / host RAM / params
"""

import argparse, os, json, time, sys, gc
from typing import Tuple, Optional
import numpy as np
import torch
from pathlib import Path

# -------- Host RAM measurement (CPU) --------
import psutil, resource

def _proc_ram_mb() -> float:
    rss = psutil.Process().memory_info().rss  # bytes
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ru_bytes = ru if sys.platform == "darwin" else ru * 1024
        peak = max(rss, ru_bytes)
    except Exception:
        peak = rss
    return peak / (1024.0 ** 2)

def parse_size(sz: str) -> Tuple[int, int]:
    w, h = sz.lower().split('x')
    return int(w), int(h)

def resolve_ckpt(ckpt_arg: str, cfg, cfg_path: Path, root: Path) -> Optional[str]:
    if ckpt_arg:
        p = Path(ckpt_arg)
        if p.is_file():
            return str(p)

        p2 = (root / ckpt_arg)
        if p2.is_file():
            return str(p2)

        p3 = (cfg_path.parent / ckpt_arg)
        if p3.is_file():
            return str(p3)

    lf = getattr(cfg, 'load_from', None) or cfg.get('load_from', None)
    if isinstance(lf, str):
        p4 = Path(lf)
        if p4.is_file():
            return str(p4)
        if (root / lf).is_file():
            return str(root / lf)
        if (cfg_path.parent / lf).is_file():
            return str(cfg_path.parent / lf)
    return None

def main():
    parser = argparse.ArgumentParser("CPU benchmark (latency / RAM / params)")
    parser.add_argument("config", help="path to config.py")
    parser.add_argument("checkpoint", help="path to checkpoint.pth")
    parser.add_argument("--sizes", nargs="+", default=["384x288"], help="WxH, e.g. 384x288 320x320")
    parser.add_argument("--bs", nargs="+", type=int, default=[1, 8], help="batch sizes to test")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=["tensor", "predict"], default="tensor")
    parser.add_argument("--out", default="", help="csv path to append/save results")
    parser.add_argument("--tag", default="", help="meta: model tag")
    parser.add_argument("--species", default="", help="meta: species")
    args = parser.parse_args()


    ROOT = Path(__file__).resolve().parents[1]
    os.chdir(ROOT)

    from mmengine import Config
    from mmpose.apis import init_model

    torch.set_grad_enabled(False)
    cfg_path = Path(args.config).resolve()
    cfg = Config.fromfile(str(cfg_path))


    ckpt = resolve_ckpt(args.checkpoint, cfg, cfg_path, ROOT)
    if ckpt is None:
        print(f"[WARN] checkpoint 未找到：{args.checkpoint}；将以 **未加载权重** 的模型跑基准。", file=sys.stderr)

    device = "cpu"

    model = init_model(str(cfg_path), ckpt, device=device)
    model.eval()

    results = []
    for sz in args.sizes:
        W, H = parse_size(sz)
        for bs in args.bs:
            imgs = [torch.randn(3, H, W) for _ in range(bs)]


            gc.collect()
            ram_base = _proc_ram_mb()
            ram_peak = ram_base


            with torch.no_grad():
                for _ in range(args.warmup):
                    try:
                        _ = model(imgs, data_samples=None, mode=args.mode)
                    except Exception:
                        _ = model(imgs, data_samples=None, mode="tensor")
                    ram_peak = max(ram_peak, _proc_ram_mb())


            times_ms = []
            with torch.no_grad():
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    try:
                        _ = model(imgs, data_samples=None, mode=args.mode)
                    except Exception:
                        _ = model(imgs, data_samples=None, mode="tensor")
                    ms = (time.perf_counter() - t0) * 1000.0
                    times_ms.append(ms)
                    ram_peak = max(ram_peak, _proc_ram_mb())

            mean_ms = float(np.mean(times_ms))
            std_ms = float(np.std(times_ms))
            fps = float(1000.0 / mean_ms * bs)
            host_ram_mb = max(0.0, ram_peak - ram_base)
            params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)

            rec = dict(
                tag=args.tag or Path(args.config).name.replace(".py", ""),
                species=args.species,
                device=device,
                size=f"{W}x{H}",
                bs=bs,
                warmup=args.warmup,
                iters=args.iters,
                mean_ms=round(mean_ms, 3),
                std_ms=round(std_ms, 3),
                fps=round(fps, 2),
                host_ram_MB=round(host_ram_mb, 1),
                params_M=params_m,
            )
            results.append(rec)
            print(json.dumps(rec, ensure_ascii=False))

            del imgs
            gc.collect()

    if args.out:
        try:
            import pandas as pd
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            df_new = pd.DataFrame(results)
            if Path(args.out).is_file():
                df_old = pd.read_csv(args.out)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.to_csv(args.out, index=False)
        except Exception as e:
            print(f"[WARN] 写 CSV 失败：{e}", file=sys.stderr)

if __name__ == "__main__":
    main()

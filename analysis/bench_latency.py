#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, time
import numpy as np
import torch

def parse_size(sz: str):
    w, h = sz.lower().split('x')
    return int(w), int(h)

def main():
    parser = argparse.ArgumentParser("Benchmark latency / memory / params")
    parser.add_argument("config", help="path to config.py")
    parser.add_argument("checkpoint", help="path to checkpoint.pth")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--sizes", nargs="+", default=["384x288"], help="WxH, e.g. 384x288 320x320")
    parser.add_argument("--bs", nargs="+", type=int, default=[1, 8], help="batch sizes to test")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=["tensor", "predict"], default="tensor")
    parser.add_argument("--out", default="", help="csv path to append/save results")
    parser.add_argument("--tag", default="", help="meta: model tag")
    parser.add_argument("--species", default="", help="meta: species")
    args = parser.parse_args()

    # Lazy imports after args parse
    from mmengine import Config
    from mmpose.apis import init_model

    cfg = Config.fromfile(args.config)
    device = args.device

    model = init_model(args.config, args.checkpoint, device=device)
    model.eval()

    results = []
    for sz in args.sizes:
        W, H = parse_size(sz)
        for bs in args.bs:
            # 准备随机输入；让 data_preprocessor 处理到正确设备
            imgs = [torch.randn(3, H, W) for _ in range(bs)]

            # CUDA 统计准备
            use_cuda = device.startswith("cuda") and torch.cuda.is_available()
            if use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            # 预热
            with torch.no_grad():
                for _ in range(args.warmup):
                    try:
                        _ = model(imgs, data_samples=None, mode=args.mode)
                    except Exception:
                        _ = model(imgs, data_samples=None, mode="tensor")
                    if use_cuda:
                        torch.cuda.synchronize()

            # 正式计时
            times_ms = []
            with torch.no_grad():
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    if use_cuda:
                        starter.record()
                    try:
                        _ = model(imgs, data_samples=None, mode=args.mode)
                    except Exception:
                        _ = model(imgs, data_samples=None, mode="tensor")
                    if use_cuda:
                        ender.record()
                        torch.cuda.synchronize()
                        ms = starter.elapsed_time(ender)
                    else:
                        ms = (time.perf_counter() - t0) * 1000.0
                    times_ms.append(ms)

            mean_ms = float(np.mean(times_ms))
            std_ms = float(np.std(times_ms))
            fps = float(1000.0 / mean_ms * bs)
            peak_mem_mb = 0.0
            if use_cuda:
                peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

            # 参数量
            params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)

            rec = dict(
                tag=args.tag or os.path.basename(args.config).replace(".py", ""),
                species=args.species,
                device=device,
                size=sz,
                bs=bs,
                warmup=args.warmup,
                iters=args.iters,
                mean_ms=round(mean_ms, 3),
                std_ms=round(std_ms, 3),
                fps=round(fps, 2),
                peak_mem_mb=round(peak_mem_mb, 1),
                params_m=params_m,
            )
            results.append(rec)
            print(json.dumps(rec, ensure_ascii=False))


    if args.out:
        try:
            import pandas as pd
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            if os.path.isfile(args.out):
                # 追加
                df_old = pd.read_csv(args.out)
                df_new = pd.concat([df_old, pd.DataFrame(results)], ignore_index=True)
            else:
                df_new = pd.DataFrame(results)
            df_new.to_csv(args.out, index=False)
        except Exception as e:
            print(f"[WARN] 写 CSV 失败：{e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练后数据与日志自检 & 曲线整理

做什么：
1) 扫描 work_dirs 下每个 run 目录，解析 *.log.json（mmengine JSONL 日志）
   - 汇总训练曲线：iter/epoch 的 loss、lr、time 等
   - 汇总验证曲线：各 coco 指标（如 coco/AP、AP.5、AP.75、AR 等）
   - 校验是否存在 best.pth/last*.pth
   - 检查日志是否包含画图必需字段（loss 与 coco/AP 至少其一）
   - 将每个 run 导出为 CSV：train_curve.csv、val_metrics.csv

2) 扫描 analysis/raw_* 下由 CocoMetric 导出的结果文件（outfile_prefix）
   - 找到 *_keypoints*.json 或 *.json 结果（COCO格式）
   - 粗查条目数、每条 keypoints 长度（是否 18*3）、score的基础统计
   - 导出一个 results_summary.csv

3) 产出一个总体的 summary.json，告诉你每个 run 是否“可画图”（has_loss / has_ap）

用法示例：
  python tools/check_post_training.py \
    --work-roots work_dirs_smoke work_dirs_smoke_cattle work_dirs_smoke_hs \
    --analysis-roots analysis/raw_smoke analysis/raw_smoke_cattle analysis/raw_smoke_hs \
    --out analysis/derived \
    --expect-kps 18

可选：
  --run-filter simcc  只检查目录名包含 'simcc' 的 run
  --strict           若缺少关键曲线则以非零码退出（默认宽松）
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from collections import defaultdict, Counter
from statistics import mean

# ---------- Utils ----------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_runs(work_root, run_filter=None):
    if not os.path.isdir(work_root):
        return []
    dirs = [d for d in sorted(glob(os.path.join(work_root, '*'))) if os.path.isdir(d)]
    if run_filter:
        dirs = [d for d in dirs if run_filter in os.path.basename(d)]
    return dirs

def latest_log_json(run_dir):
    """mmengine 默认产生日志：YYYYMMDD_HHMMSS.log.json"""
    cands = sorted(glob(os.path.join(run_dir, '*.log.json')))
    return cands[-1] if cands else None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def sanitize_key(k):
    """CSV 列名安全化（把 / 和 空格 换成 _）"""
    return k.replace('/', '_').replace(' ', '').replace('.', '')

# ---------- Parse mmengine JSONL log ----------

def parse_log_json(log_path):
    """
    返回:
      train_rows: [{'epoch':..,'iter':..,'loss':..,'lr':.., ...}, ...]
      val_rows  : [{'epoch':..,'coco/AP':.., 'coco/AP .5':.., ...}, ...]
    """
    train_rows, val_rows = [], []
    if not log_path or not os.path.isfile(log_path):
        return train_rows, val_rows

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # 常见字段
            mode = obj.get('mode', None)  # 'train' / 'val' / 'test' / None
            epoch = obj.get('epoch', obj.get('epoch_idx', None))
            if isinstance(epoch, str):
                # 例如 "1/300"
                try:
                    epoch = int(str(epoch).split('/')[0])
                except Exception:
                    pass

            row = {'epoch': epoch}
            # 把所有数值型字段都捞一下
            for k, v in obj.items():
                if k in ('epoch', 'epoch_idx', 'mode', 'msg', 'time', 'data_time', 'memory', 'eta', 'lr', 'iter', 'loss'):
                    pass
                # 把 coco/AP 这类 metric 捞出来
            # iter/lr/loss/time
            if 'iter' in obj:
                row['iter'] = obj['iter']
            if 'lr' in obj:
                fl = safe_float(obj['lr'])
                if fl is not None:
                    row['lr'] = fl
            if 'loss' in obj:
                fl = safe_float(obj['loss'])
                if fl is not None:
                    row['loss'] = fl
            if 'time' in obj:
                fl = safe_float(obj['time'])
                if fl is not None:
                    row['time'] = fl

            # 收集所有 float-like 的 metric（含 coco/*）
            for k, v in obj.items():
                if k in ('mode', 'epoch', 'epoch_idx', 'iter', 'lr', 'loss', 'time', 'data_time', 'eta', 'memory', 'msg'):
                    continue
                fl = safe_float(v)
                if fl is not None:
                    # mmengine 里 coco 指标通常是 'coco/AP', 'coco/AP .5', ...
                    row[k] = fl

            if mode == 'train':
                if len(row) > 1:
                    train_rows.append(row)
            elif mode == 'val':
                # 只记录 metric（无需 iter/loss）
                val_rows.append(row)

    return train_rows, val_rows

# ---------- Save CSV ----------

def save_csv(rows, out_csv):
    if not rows:
        return 0
    # 汇总所有列
    cols = set()
    for r in rows:
        cols = cols | set(r.keys())
    cols = ['epoch'] + sorted([c for c in cols if c != 'epoch'])
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, '')
                if isinstance(v, float):
                    vals.append(f'{v:.6g}')
                else:
                    vals.append(str(v))
            f.write(','.join(vals) + '\n')
    return len(rows)

# ---------- Scan analysis results (COCO json) ----------

def scan_analysis_dir(ana_root, expect_kps=18):
    """
    返回一个列表，每个元素形如：
    {
      'path': json_path,
      'num_preds': N,
      'avg_score': ...,
      'bad_kpt_len': count
    }
    """
    out = []
    if not os.path.isdir(ana_root):
        return out
    # 常见文件名：<prefix>_keypoints.json / <prefix>.keypoints.json / 任意 *.json
    jsons = sorted(glob(os.path.join(ana_root, '**', '*.json'), recursive=True))
    for jp in jsons:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            continue
        # 可能是 metrics 摘要（dict），也可能是 COCO 结果（list）
        if isinstance(obj, list):
            # COCO keypoints 结果应该是 list[dict]
            n = len(obj)
            if n == 0:
                out.append({'path': jp, 'num_preds': 0, 'avg_score': 0.0, 'bad_kpt_len': 0})
                continue
            scores, bad = [], 0
            for d in obj:
                kp = d.get('keypoints', [])
                if len(kp) != expect_kps * 3:
                    bad += 1
                sc = d.get('score', None)
                if isinstance(sc, (int, float)):
                    scores.append(float(sc))
            out.append({
                'path': jp,
                'num_preds': n,
                'avg_score': (mean(scores) if scores else 0.0),
                'bad_kpt_len': bad
            })
        # 如果是 dict，可能是 coco_eval 的指标摘要（可忽略或另存）
    return out

def save_results_summary(rows, out_csv):
    if not rows:
        return 0
    cols = ['path', 'num_preds', 'avg_score', 'bad_kpt_len']
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join([str(r.get(c, '')) for c in cols]) + '\n')
    return len(rows)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-roots', nargs='*', default=['work_dirs_smoke'], help='包含 run 子目录的根')
    ap.add_argument('--analysis-roots', nargs='*', default=['analysis/raw_smoke'], help='保存结果 json 的根')
    ap.add_argument('--out', default='analysis/derived', help='导出的 CSV / 总结输出目录')
    ap.add_argument('--run-filter', default='', help='仅检查目录名包含该子串的 run')
    ap.add_argument('--expect-kps', type=int, default=18)
    ap.add_argument('--strict', action='store_true', help='缺关键曲线则以非零退出')
    args = ap.parse_args()

    ensure_dir(args.out)

    # 1) 处理 work_dirs
    summary = []
    missing_key_runs = 0

    for wr in args.work_roots:
        runs = find_runs(wr, args.run_filter)
        print(f"[SCAN] {wr} 发现 {len(runs)} 个 run")
        for run in runs:
            run_name = os.path.basename(run)
            log_json = latest_log_json(run)
            run_out_dir = os.path.join(args.out, run_name)
            ensure_dir(run_out_dir)

            train_rows, val_rows = parse_log_json(log_json) if log_json else ([], [])
            train_csv = os.path.join(run_out_dir, 'train_curve.csv')
            val_csv   = os.path.join(run_out_dir, 'val_metrics.csv')

            n_train = save_csv(train_rows, train_csv) if train_rows else 0
            n_val   = save_csv(val_rows,   val_csv)   if val_rows   else 0

            # checkpoint 存在性
            ckpts = sorted(glob(os.path.join(run, '*.pth')))
            has_best = any(os.path.basename(p) == 'best.pth' for p in ckpts)
            has_any  = len(ckpts) > 0

            # 关键字段是否齐全
            has_loss = any('loss' in r for r in train_rows)
            # val 常见 key（不同版本可能为 'coco/AP' 或 'coco/AP (bbox)'）
            keys = set().union(*[set(r.keys()) for r in val_rows]) if val_rows else set()
            has_ap = any(k.startswith('coco/AP') or k == 'coco/AP' for k in keys)

            if not (has_loss or has_ap):
                missing_key_runs += 1

            summary.append({
                'run_dir': run,
                'log_json': log_json or '',
                'train_csv': train_csv if n_train else '',
                'val_csv': val_csv if n_val else '',
                'n_train_rows': n_train,
                'n_val_rows': n_val,
                'has_best_ckpt': has_best,
                'ckpt_count': len(ckpts),
                'has_loss_curve': has_loss,
                'has_coco_ap': has_ap
            })
            print(f"  - {run_name}: loss_rows={n_train}, val_rows={n_val}, best={has_best}, ckpts={len(ckpts)}, has_loss={has_loss}, has_AP={has_ap}")

    # 2) 处理 analysis/raw_*
    all_results = []
    for ar in args.analysis_roots:
        rows = scan_analysis_dir(ar, expect_kps=args.expect_kps)
        out_csv = os.path.join(args.out, f'{os.path.basename(ar)}_results_summary.csv')
        n = save_results_summary(rows, out_csv)
        print(f"[RESULTS] {ar}: files={len(rows)}, saved={n} -> {out_csv}")
        all_results.extend(rows)

    # 3) 写出总体 summary.json
    summary_path = os.path.join(args.out, 'postcheck_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'runs': summary,
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] 汇总写入：{summary_path}")

    if args.strict and missing_key_runs > 0:
        print(f"[FAIL] 有 {missing_key_runs} 个 run 缺少作图关键曲线（loss/AP）。")
        sys.exit(2)
    sys.exit(0)


if __name__ == '__main__':
    main()

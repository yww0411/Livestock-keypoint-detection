#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 AP-10K 的 42 张羊标注，切成“固定测试集 + 训练池”，并在训练池里按 (K, seed) 采样 few-shot 训练子集。

目录：
  ROOT = hy-tmp/keypoint-detection/
  公共数据：ROOT/data_process/test_data/
    - annotations/test_annotations.coco.json   # 42 张全集
    - images/

输出：
  - ROOT/data_process/test_data/annotations/fewshot/test_fixed.json
  - ROOT/data_process/test_data/annotations/fewshot/pool.json
  - ROOT/data_process/test_data/annotations/fewshot/train_{K}_s{seed}.json  (K ∈ shots, seed ∈ seeds)
"""

from __future__ import annotations
from pathlib import Path
import json, random, argparse

ROOT = Path(__file__).resolve().parents[2]   # .../hy-tmp/keypoint-detection
ANN_ALL = ROOT/'data_process'/'test_data'/'annotations'/'test_annotations.coco.json'
OUT_DIR = ROOT/'data_process'/'test_data'/'annotations'/'fewshot'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_index(coco):
    imgs = {im['id']: im for im in coco['images']}
    img2anns = {}
    for a in coco['annotations']:
        img2anns.setdefault(a['image_id'], []).append(a)
    return imgs, img2anns

def dump_subset(img_ids, imgs, img2anns, cats, out_path: Path):
    sub = {
        'images': [imgs[i] for i in img_ids],
        'annotations': [a for i in img_ids for a in img2anns.get(i, [])],
        'categories': cats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(sub, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', default=str(ANN_ALL), help='42 张全集 COCO（test_annotations.coco.json）')
    ap.add_argument('--num-test', type=int, default=16, help='固定测试集大小（默认 16）')
    ap.add_argument('--shots', type=int, nargs='+', default=[10, 20, 30], help='few-shot 档位')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1337, 2029, 3407], help='随机种子集合')
    ap.add_argument('--global-seed', type=int, default=114514, help='固定测试集的全局种子（保证 test 固定不随 seed 变化）')
    args = ap.parse_args()

    coco = json.load(open(args.ann, 'r'))
    imgs, img2anns = build_index(coco)
    all_ids = [im['id'] for im in coco['images']]
    assert len(all_ids) >= args.num_test, 'num-test 不能超过总图像数'

    # 1) 固定测试集（与 few-shot 种子无关）
    rng = random.Random(args.global_seed)
    test_fixed = sorted(rng.sample(all_ids, k=args.num_test))
    pool_ids   = sorted([i for i in all_ids if i not in set(test_fixed)])

    dump_subset(test_fixed, imgs, img2anns, coco['categories'], OUT_DIR/'test_fixed.json')
    dump_subset(pool_ids,   imgs, img2anns, coco['categories'], OUT_DIR/'pool.json')

    # 2) few-shot 训练子集（在训练池里采样）
    for seed in args.seeds:
        rng = random.Random(seed)
        pool_shuffled = pool_ids[:]
        rng.shuffle(pool_shuffled)
        for K in args.shots:
            assert K <= len(pool_ids), f'K={K} 大于训练池大小 {len(pool_ids)}'
            train_ids = sorted(pool_shuffled[:K])
            out = OUT_DIR/f'train_{K}_s{seed}.json'
            dump_subset(train_ids, imgs, img2anns, coco['categories'], out)

    print(f'[OK] 固定测试集: {OUT_DIR/"test_fixed.json"}  训练池: {OUT_DIR/"pool.json"}')
    print(f'[OK] few-shot 训练子集已生成到 {OUT_DIR}/train_<K>_s<seed>.json')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, random, collections, glob, re
from PIL import Image, ImageDraw, ImageFont

random.seed(2025)

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_ann_std(root, split):
    d = os.path.join(root, 'annotations')
    pats = [f'{split}_annotations.coco.json', f'{split}.json', f'{split}_*.json', f'*{split}*.json']
    cands = []
    for pat in pats:
        cands += glob.glob(os.path.join(d, pat))
    cands = sorted(set(cands), key=lambda x: (len(os.path.basename(x)), x))
    return cands[0] if cands else None

def resolve_img_path(root, file_name, split_hint=None):
    # 优先 images/{split}/file_name，找不到退回 root/file_name
    if split_hint in ('train', 'val', 'test'):
        p1 = os.path.join(root, 'images', split_hint, file_name)
        if os.path.exists(p1): return p1
    p2 = os.path.join(root, file_name)
    return p2

def choose_kp_names(coco):
    cats = {c.get('id'): c for c in coco.get('categories', [])}
    cnt = collections.Counter(a.get('category_id') for a in coco.get('annotations', []))
    kp = []
    if cnt:
        cid, _ = cnt.most_common(1)[0]
        kp = cats.get(cid, {}).get('keypoints', [])
    if not kp:
        for c in coco.get('categories', []):
            if c.get('keypoints'):
                kp = c['keypoints']; break
    return kp

def vis_one(image_path, anns, kp_names, out_path, draw_bbox=True):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    # 字体（默认字体足够；失败则不写字）
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for a in anns:
        # bbox
        if draw_bbox and a.get('bbox'):
            x, y, w, h = a['bbox']
            draw.rectangle([x, y, x+w, y+h], outline=(0,255,255), width=2)

        kps = a.get('keypoints', [])
        if len(kps) != 54:
            continue
        for i in range(min(18, len(kp_names))):
            x, y, v = kps[3*i:3*i+3]
            if v == 2: color = (0,200,0)
            elif v == 1: color = (255,165,0)
            else: color = (255,0,0)
            r = 3
            draw.ellipse((x-r, y-r, x+r, y+r), outline=color, fill=color, width=2)
            if font:
                draw.text((x+4, y-6), f'{i}', fill=color, font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, quality=90)

def main():
    ap = argparse.ArgumentParser(description='Quick visualize COCO keypoints')
    ap.add_argument('--ann', required=True, help='标注 json 路径')
    ap.add_argument('--root', required=True, help='图像根目录（物种目录）')
    ap.add_argument('--split', choices=['train','val','test'], default=None, help='若为马/羊，传入对应 split 以补前缀')
    ap.add_argument('--out', default='quick_vis', help='输出目录')
    ap.add_argument('--n', type=int, default=12, help='随机抽样可视化数量')
    ap.add_argument('--ids', type=int, nargs='*', help='指定 image_id 列表（覆盖 --n）')
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--no-bbox', action='store_true', help='不画 bbox')
    args = ap.parse_args()
    random.seed(args.seed)

    coco = load_json(args.ann)
    kp_names = choose_kp_names(coco)
    by_img = collections.defaultdict(list)
    for a in coco.get('annotations', []):
        by_img[a['image_id']].append(a)

    images = coco.get('images', [])
    # 选择要可视化的图
    if args.ids:
        sel = [im for im in images if im['id'] in set(args.ids)]
    else:
        sel = random.sample(images, min(args.n, len(images)))

    print(f'[INFO] keypoint names ({len(kp_names)}): {kp_names}')
    print(f'[INFO] visualize {len(sel)} images -> {args.out}')

    cnt_ok = 0
    for im in sel:
        img_path = resolve_img_path(args.root, im.get('file_name',''), args.split)
        out_path = os.path.join(args.out, f"{os.path.basename(im.get('file_name',''))}_vis.jpg")
        anns = by_img.get(im['id'], [])
        if not os.path.exists(img_path):
            print(f'[WARN] not found: {img_path}')
            continue
        vis_one(img_path, anns, kp_names, out_path, draw_bbox=not args.no_bbox)
        print(f'[OK] {out_path}')
        cnt_ok += 1

    print(f'[DONE] saved {cnt_ok} visualizations to {args.out}')

if __name__ == '__main__':
    main()

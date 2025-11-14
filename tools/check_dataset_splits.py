#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据自检
- cattle: 检查 P1 三折（p1A/B/C）的 train/val/test 互斥、缺图、可见性分布、示例打印
- horse/sheep: 自动匹配 *_annotations.coco.json，按 split 使用 images/{train|val|test}/ 作为图片前缀
- 类别选择：根据 annotations 中主流 category_id 选择对应的 categories[*] 读取 keypoints
- 泄漏检测：用 “规范化文件名 + 尺寸” 作为 UID（避免 image_id 伪冲突）
用法：
  python tools/check_dataset_splits.py --root /hy-tmp/keypoint-detection/data_process --sample 50
"""
import os, json, argparse, random, collections, glob, re
random.seed(2025)

CANONICAL = [
    'mouth','eye','ear','neck','shoulder','chest','hip','tail',
    'elbow','l_fore_wrist','l_fore_foot','r_fore_wrist','r_fore_foot',
    'hind_knee','l_hind_hock','l_hind_foot','r_hind_hock','r_hind_foot'
]

def load_json(p): 
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

def find_ann_std(root, split):
    d = os.path.join(root, 'annotations')
    pats = [f'{split}_annotations.coco.json', f'{split}.json', f'{split}_*.json', f'*{split}*.json']
    cands = []
    for pat in pats: cands += glob.glob(os.path.join(d, pat))
    cands = sorted(set(cands), key=lambda x: (len(os.path.basename(x)), x))
    return cands[0] if cands else None

def norm_uid(file_name, w=None, h=None):
    fn = file_name.replace('\\','/')
    fn = re.sub(r'(^|/)images/(train|val|test)/', 'images/', fn)  # 去掉 split 前缀
    base = os.path.basename(fn)
    wh = f'{w}x{h}' if (w and h) else ''
    return f'{base}|{wh}'

def choose_kp_names(coco):
    """按 annotations 中最常见的 category_id 选择对应的 keypoints 名单；找不到则返回空列表"""
    cats = {c.get('id'): c for c in coco.get('categories', [])}
    cnt = collections.Counter(a.get('category_id') for a in coco.get('annotations', []))
    if not cnt: return []
    cid, _ = cnt.most_common(1)[0]
    kp = cats.get(cid, {}).get('keypoints', [])
    # 兜底：若该类没 keypoints，尝试第一条有 keypoints 的类别
    if not kp:
        for c in coco.get('categories', []):
            if c.get('keypoints'): kp = c['keypoints']; break
    return kp

def resolve_img_path(root, file_name, split_hint=None):
    """优先尝试 root/images/{split}/file_name；找不到再尝试 root/file_name"""
    if split_hint in ('train','val','test'):
        p1 = os.path.join(root, 'images', split_hint, file_name)
        if os.path.exists(p1): return p1
    p2 = os.path.join(root, file_name)
    return p2

def print_first_pairs(name, coco, kp_names, topn=1):
    if not coco.get('annotations') or not coco.get('images') or not kp_names: return
    by_img = collections.defaultdict(list)
    for a in coco['annotations']: by_img[a['image_id']].append(a)
    shown = 0
    for im in coco['images']:
        if im['id'] not in by_img: continue
        a = by_img[im['id']][0]
        kps = a.get('keypoints', [])
        if len(kps) != 54: continue
        print(f"[{name}] 示例 image_id={im['id']} file={im.get('file_name','')}")
        for i, n in enumerate(kp_names[:18]):
            x, y, v = kps[3*i:3*i+3]
            print(f"  {i:02d} {n:16s} -> ({x:.1f},{y:.1f}) v={int(v)}")
        shown += 1
        if shown >= topn: break

def split_report(name, coco, root, split_hint=None, sample_n=30):
    """统计/抽样查图/可见性分布/内部重复/示例打印"""
    ok = True
    imgs = coco.get('images', [])
    anns = coco.get('annotations', [])
    kp_names = choose_kp_names(coco)

    # keypoints 名单提示（只警告，不阻塞）
    if not kp_names:
        print(f"[{name}] WARN: 找不到 categories.keypoints（可能类别未携带 keypoints），训练将以 metainfo 为准")
    elif len(kp_names) != 18:
        print(f"[{name}] WARN: categories.keypoints 数量={len(kp_names)}，期望=18")
    elif [x.lower() for x in kp_names] != CANONICAL:
        print(f"[{name}] WARN: keypoint 名称顺序与 canonical 不完全一致（不影响训练，但建议统一）")

    # 可见性分布
    vis_cnt = collections.Counter()
    bad = 0
    for a in anns:
        kps = a.get('keypoints', [])
        if len(kps) != 54: bad += 1
        for i in range(2, len(kps), 3):
            vis_cnt[kps[i]] += 1
    tot = sum(vis_cnt.values())
    if tot == 0:
        print(f"[{name}] ERROR: 没有关键点"); ok = False
    else:
        pct = ", ".join([f"v={k}:{100*v/tot:.1f}%" for k,v in sorted(vis_cnt.items())])
        print(f"[{name}] v 分布: {dict(vis_cnt)} | {pct}")
    if bad: print(f"[{name}] WARN: 有 {bad} 条 annotation 的 keypoints 长度!=54")

    # split 内部重复
    uids = [norm_uid(im.get('file_name',''), im.get('width'), im.get('height')) for im in imgs]
    dup = [u for u,c in collections.Counter(uids).items() if c>1]
    if dup: print(f"[{name}] WARN: split 内重复 {len(dup)}"); 
    else:   print(f"[{name}] split 内无重复 ✅")

    # 抽样查图（考虑 split 前缀）
    miss = []
    picks = random.sample(imgs, min(sample_n, len(imgs)))
    for im in picks:
        f = im.get('file_name','')
        p = resolve_img_path(root, f, split_hint)
        if not os.path.exists(p): miss.append(f)
    if miss:
        print(f"[{name}] ERROR: 抽样缺图 {len(miss)}/{len(picks)}，示例: {miss[:5]}"); ok = False
    else:
        print(f"[{name}] 抽样图片均存在 ✅")

    # 打印一张样例：(name,x,y,v) 映射，方便肉眼核对名称对齐
    print_first_pairs(name, coco, kp_names, topn=1)
    return ok, set(uids)

def check_disjoint(tag, a, b, c):
    ok = True
    for nm, s in [('train∩val',a&b),('train∩test',a&c),('val∩test',b&c)]:
        if s:
            print(f"[{tag}] ERROR: {nm} 非空，示例: {list(s)[:5]}"); ok=False
    if ok: print(f"[{tag}] 三个 split 互斥 ✅")
    return ok

def check_cattle(root, fold, sample_n):
    ad = os.path.join(root, 'annotations_p1')
    tr = load_json(os.path.join(ad, f'cattle_{fold}_train.json'))
    va = load_json(os.path.join(ad, f'cattle_{fold}_val.json'))
    te = load_json(os.path.join(ad, f'cattle_{fold}_test.json'))
    print(f"[cattle-{fold}] 数量: train={len(tr['images'])} val={len(va['images'])} test={len(te['images'])}")
    # cattle 的 file_name 已带 A/B/C/images/... 前缀，所以不传 split_hint
    ok1,u1 = split_report(f'cattle-{fold}-train', tr, root, None, sample_n)
    ok2,u2 = split_report(f'cattle-{fold}-val',   va, root, None, sample_n)
    ok3,u3 = split_report(f'cattle-{fold}-test',  te, root, None, sample_n)
    ok4     = check_disjoint(f'cattle-{fold}', u1,u2,u3)
    return ok1 and ok2 and ok3 and ok4

def check_std(root, species, sample_n):
    tr_p, va_p, te_p = [find_ann_std(root, s) for s in ('train','val','test')]
    if not (tr_p and va_p and te_p):
        print(f"[{species}] ERROR: 未找到完整的 train/val/test 标注"); return False
    jtr, jva, jte = load_json(tr_p), load_json(va_p), load_json(te_p)
    print(f"[{species}] 标注：{os.path.basename(tr_p)}, {os.path.basename(va_p)}, {os.path.basename(te_p)}")
    print(f"[{species}] 数量: train={len(jtr['images'])} val={len(jva['images'])} test={len(jte['images'])}")
    ok1,u1 = split_report(f'{species}-train', jtr, root, 'train', sample_n)
    ok2,u2 = split_report(f'{species}-val',   jva, root, 'val',   sample_n)
    ok3,u3 = split_report(f'{species}-test',  jte, root, 'test',  sample_n)
    ok4     = check_disjoint(species, u1,u2,u3)
    return ok1 and ok2 and ok3 and ok4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='data_process 目录')
    ap.add_argument('--sample', type=int, default=50)
    args = ap.parse_args()

    base = args.root
    all_ok = True
    # cattle：P1 三折
    for fold in ('p1A','p1B','p1C'):
        print("\n"+"="*12, f"cattle {fold}", "="*12)
        all_ok &= check_cattle(os.path.join(base,'cattle'), fold, args.sample)

    # horse/sheep
    print("\n"+"="*12, "horse", "="*12)
    all_ok &= check_std(os.path.join(base,'horse'), 'horse', args.sample)
    print("\n"+"="*12, "sheep", "="*12)
    all_ok &= check_std(os.path.join(base,'sheep'), 'sheep', args.sample)

    print("\n==== 总结 ====")
    print("✅ ALL PASS" if all_ok else "❌ 存在问题，见上面日志")
    raise SystemExit(0 if all_ok else 1)

if __name__ == '__main__':
    main()

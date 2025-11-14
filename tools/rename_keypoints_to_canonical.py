#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 COCO 标注中的 keypoints 名称与顺序统一为 canonical，并重排每条 annotation 的 keypoints。
- 支持马/羊/牛；对含 keypoints 的所有 category 生效（按 annotations 中用到的 category_id 处理）
- 清空 categories[*].skeleton，避免索引错位
- 修正 annotation.num_keypoints = count(v>=1)
- 输出到 *_canon.json，或用 --inplace 覆盖源文件

用法:
  python tools/rename_keypoints_to_canonical.py --in /path/to/xxx.json
  python tools/rename_keypoints_to_canonical.py --in /path/to/xxx.json --inplace
"""
import os, json, argparse, sys
from collections import Counter

CANONICAL = [
    'mouth','eye','ear','neck','shoulder','chest','hip','tail',
    'elbow','l_fore_wrist','l_fore_foot','r_fore_wrist','r_fore_foot',
    'hind_knee','l_hind_hock','l_hind_foot','r_hind_hock','r_hind_foot'
]

# 别名映射（小写比较）
ALIAS = {
    # 直接同名
    'mouth':'mouth','eye':'eye','ear':'ear','neck':'neck',
    'shoulder':'shoulder','chest':'chest','hip':'hip','tail':'tail',
    'elbow':'elbow','l_fore_wrist':'l_fore_wrist','l_fore_foot':'l_fore_foot',
    'r_fore_wrist':'r_fore_wrist','r_fore_foot':'r_fore_foot',
    'hind_knee':'hind_knee','l_hind_hock':'l_hind_hock','l_hind_foot':'l_hind_foot',
    'r_hind_hock':'r_hind_hock','r_hind_foot':'r_hind_foot',
    # 你的短名
    'body1':'shoulder','body2':'chest','body3':'hip','body4':'tail',
    'f':'elbow','b':'hind_knee',
    'fl1':'l_fore_wrist','fl2':'l_fore_foot',
    'fr1':'r_fore_wrist','fr2':'r_fore_foot',
    'bl1':'l_hind_hock','bl2':'l_hind_foot',
    'br1':'r_hind_hock','br2':'r_hind_foot',
    # 可能的长名（容错）
    'front':'elbow','behind':'hind_knee',
    'front_left1':'l_fore_wrist','front_left2':'l_fore_foot',
    'front_right1':'r_fore_wrist','front_right2':'r_fore_foot',
    'behind_left1':'l_hind_hock','behind_left2':'l_hind_foot',
    'behind_right1':'r_hind_hock','behind_right2':'r_hind_foot',
}

def load_json(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

def save_json(obj, p):
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False)

def build_perm(old_names):
    """
    由旧名字列表 -> 计算到 canonical 的置换索引 perm（len=18）：
    new[j] = old[perm[j]]
    """
    old_l = [n.lower() for n in old_names]
    # 旧名先归一化到 canonical 名字
    mapped = []
    for n in old_l:
        if n not in ALIAS:
            raise ValueError(f"未识别的关键点名称: '{n}'. 请在 ALIAS 中补充映射。")
        mapped.append(ALIAS[n])

    # 每个 canonical 名应在 mapped 中恰好出现一次
    idxs = []
    used = set()
    for cn in CANONICAL:
        try:
            i = next(k for k, v in enumerate(mapped) if v == cn and k not in used)
        except StopIteration:
            raise ValueError(f"旧顺序中缺少 '{cn}'，无法重排。旧→映射后列表: {mapped}")
        used.add(i)
        idxs.append(i)
    if len(set(idxs)) != 18:
        raise ValueError("映射出现重复，请检查 ALIAS 与旧名称。")
    return idxs  # len=18

def reorder_kps_vec(vec, perm):
    """按 perm 重排 18×3 的扁平 keypoints 向量"""
    if len(vec) != 54: return vec  # 保持原样，但上层会给出 WARNING
    out = [0.0]*54
    for j, i in enumerate(perm):
        out[3*j:3*j+3] = vec[3*i:3*i+3]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='输入 COCO 标注 json 路径')
    ap.add_argument('--inplace', action='store_true', help='就地覆盖（否则写 *_canon.json）')
    args = ap.parse_args()

    j = load_json(args.inp)

    # 找到含 keypoints 的类别，并按 annotations 中使用频次排序
    cat_by_id = {c.get('id'): c for c in j.get('categories', [])}
    used_counts = Counter(a.get('category_id') for a in j.get('annotations', []))
    target_cats = [cid for cid,_ in used_counts.most_common() if cid in cat_by_id and cat_by_id[cid].get('keypoints')]

    if not target_cats:
        print("[WARN] categories 中没有携带 keypoints 的类；仅写入/覆盖第一条类的 keypoints 顺序。")
        # 若 categories 为空，补一条
        if not j.get('categories'):
            j['categories'] = [{'id': 1, 'name': 'animal', 'supercategory': 'animal'}]
        target_cats = [j['categories'][0]['id']]
        cat_by_id = {c.get('id'): c for c in j.get('categories', [])}
        cat_by_id[target_cats[0]]['keypoints'] = CANONICAL[:]  # 临时给出

    # 以第一个目标类的旧顺序推导 perm
    old_names = cat_by_id[target_cats[0]].get('keypoints', [])
    if len(old_names) != 18:
        print(f"[INFO] 旧 keypoints 名单长度为 {len(old_names)}，仍尝试按别名重建置换。")
    perm = build_perm(old_names)

    # 重排所有含 keypoints 的类别：名称改为 canonical、skeleton 清空
    for cid in cat_by_id:
        if cat_by_id[cid].get('keypoints') is not None:
            cat_by_id[cid]['keypoints'] = CANONICAL[:]
            cat_by_id[cid]['skeleton'] = []

    # 重排 annotations 中对应类别的 keypoints
    bad = 0
    for a in j.get('annotations', []):
        if a.get('category_id') in cat_by_id:  # 该类存在
            kps = a.get('keypoints', [])
            if len(kps) != 54: bad += 1
            a['keypoints'] = reorder_kps_vec(kps, perm)
            # 修正 num_keypoints（v>=1 计数）
            a['num_keypoints'] = int(sum(1 for i in range(2, len(a['keypoints']), 3) if a['keypoints'][i] >= 1))

    if bad:
        print(f"[WARN] 有 {bad} 条 annotation 的 keypoints 长度不是 54，已跳过重排（其余已重排）。")

    # 写出
    out = args.inp if args.inplace else os.path.splitext(args.inp)[0] + "_canon.json"
    save_json(j, out)
    print(f"[OK] 已统一 keypoint 名称与顺序，写出：{out}")

if __name__ == '__main__':
    main()

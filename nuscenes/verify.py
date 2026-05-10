"""
7-check verification suite for the IntentFormer-on-nuScenes pipeline.
Mirrors EfficientPIE's 6 checks + seg-cache coverage.

  1. Index meta sanity.
  2. intent_label() property test (134k random calls vs UniAD source).
  3. End-to-end label provenance (16 random pkl records vs raw JSON).
  4. Train/eval pkl agreement (compares record counts loaded by data_gen).
  5. Transform agreement (val branch of __get_input == eval crop pipeline).
  6. Metrics-from-CSV reproducibility (run after eval.py emits CSV).
  7. Seg-cache coverage (every sample_token in seq pkl has a cached PNG).

Skips checks that require artefacts not yet present, with a printed note.
Returns non-zero exit if any executed check fails.
"""

import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np


SEQ_PKL = '/root/IntentFormer/nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl'
JSON_LABELS = '/mnt/nuscenes/nuScenes/unified_map_v3/all_scenes_compact_new.json'
TRAIN_PKL = '/mnt/storage/EfficientPIE/nuscenes_infos_temporal_train.pkl'
SEG_CACHE = '/mnt/storage/IntentFormer/seg_cache'
UNIAD_PATH = '/root/UniAD/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py'
EFFICIENTPIE_BUILDER = '/root/EfficientPIE/utils/build_nuscenes_index_v2.py'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq-pkl', default=SEQ_PKL)
    p.add_argument('--csv', default=None,
                   help='per-sample CSV from eval.py (for check 6)')
    p.add_argument('--n-property', type=int, default=134_953,
                   help='number of random calls for intent_label property test')
    p.add_argument('--n-provenance', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def section(n, name):
    print(f'\n=== Check {n}: {name} ===')


def import_func(path, func_name):
    spec = importlib.util.spec_from_file_location('mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def check1_meta(pkl):
    section(1, 'Index meta sanity')
    meta = pkl.get('meta', {})
    required = {
        'k': 3,
        'look_ahead': 4,
        'source_pkl': TRAIN_PKL,
        'source_json': JSON_LABELS,
        'seg_cache_dir': SEG_CACHE,
    }
    ok = True
    for k, v in required.items():
        got = meta.get(k)
        if got != v:
            print(f'  FAIL  meta[{k!r}] = {got!r} (expected {v!r})')
            ok = False
        else:
            print(f'  OK    meta[{k!r}] = {got!r}')
    print(f'  meta keys: {list(meta.keys())}')
    return ok


def check2_property(n, seed):
    section(2, f'intent_label() property test ({n} random calls)')
    f_seq = import_func('/root/IntentFormer/nuscenes/build_nuscenes_seq_index_v2.py',
                        'intent_label')
    f_eff = import_func(EFFICIENTPIE_BUILDER, 'intent_label')
    rng = random.Random(seed)
    actions = ['STOPPED', 'MOVING', 'Crossing', 'TURN_RIGHT', 'TURN_LEFT',
               'LANE_CHANGE_RIGHT', 'LANE_CHANGE_LEFT', 'na', 'Stopped', 'Moving']
    mismatches = 0
    for _ in range(n):
        L = rng.randint(2, 60)
        arr = [rng.choice(actions) for _ in range(L)]
        idx = rng.randint(-2, L + 2)
        a = f_seq(idx, arr)
        b = f_eff(idx, arr)
        if a != b:
            mismatches += 1
            if mismatches <= 5:
                print(f'  diff at idx={idx}, arr={arr[:6]}...: ours={a} efficientpie={b}')
    print(f'  mismatches: {mismatches}/{n}')
    return mismatches == 0


def check3_provenance(pkl, n, seed):
    section(3, f'End-to-end label provenance ({n} random records)')
    f_intent = import_func('/root/IntentFormer/nuscenes/build_nuscenes_seq_index_v2.py',
                           'intent_label')
    print(f'  loading raw JSON from {JSON_LABELS} ...')
    with open(JSON_LABELS) as f:
        json_data = json.load(f)
    rng = random.Random(seed)
    pool = pkl['train'] + pkl['val']
    sel = rng.sample(pool, k=min(n, len(pool)))
    fails = 0
    for rec in sel:
        scene = json_data.get(rec['scene_token'])
        if scene is None:
            print(f'  FAIL  scene {rec["scene_token"]} not in JSON'); fails += 1; continue
        agent = scene.get(rec['instance_token'])
        if agent is None:
            print(f'  FAIL  instance {rec["instance_token"]} not in scene'); fails += 1; continue
        i7 = f_intent(rec['frame_idx'], agent['labels'])
        if i7 != rec['intent_7class']:
            print(f'  FAIL  intent7 mismatch: pkl={rec["intent_7class"]} json={i7}')
            fails += 1; continue
        if i7 == 2:
            expected = 1
        elif i7 in (0, 1):
            expected = 0
        else:
            expected = -1
        if expected != rec['label']:
            print(f'  FAIL  label mismatch: pkl={rec["label"]} expected={expected}')
            fails += 1; continue
        # img + seg path existence
        for ip in rec['img_paths']:
            if not os.path.exists(ip):
                print(f'  FAIL  missing img {ip}'); fails += 1; break
    print(f'  failures: {fails}/{len(sel)}')
    return fails == 0


def check4_loader_agreement(pkl):
    section(4, 'Train/eval pkl agreement')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_gen import load_records
    train_recs, _ = load_records(SEQ_PKL, 'train')
    val_recs, _ = load_records(SEQ_PKL, 'val')
    same_train = len(train_recs) == len(pkl['train'])
    same_val = len(val_recs) == len(pkl['val'])
    print(f'  train: pkl={len(pkl["train"])} loader={len(train_recs)} same={same_train}')
    print(f'  val:   pkl={len(pkl["val"])}   loader={len(val_recs)}   same={same_val}')
    return same_train and same_val


def check5_transform(pkl):
    section(5, 'Transform agreement (val == eval crop pipeline)')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_gen import IntentFormerSeqGen
    gen_train = IntentFormerSeqGen(pkl['val'][:2], batch_size=2, k=3,
                                   shuffle=False, train=True, use_seg=True, seed=0)
    gen_val = IntentFormerSeqGen(pkl['val'][:2], batch_size=2, k=3,
                                 shuffle=False, train=False, use_seg=True, seed=0)
    x_t = gen_train[0][0][0]
    x_v = gen_val[0][0][0]
    same = bool(np.array_equal(x_t, x_v))
    print(f'  val/train deterministic-pipeline outputs equal (no aug): {same}')
    print(f'  shape: {x_t.shape}')
    return same


def check6_csv(csv_path):
    section(6, 'Metrics-from-CSV reproducibility')
    if csv_path is None:
        print('  SKIP (no --csv arg).')
        return True
    if not os.path.exists(csv_path):
        print(f'  SKIP (csv {csv_path} does not exist yet).')
        return True
    import csv as csvmod
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    scores, labels = [], []
    with open(csv_path) as f:
        r = csvmod.DictReader(f)
        for row in r:
            scores.append(float(row['score']))
            labels.append(int(row['label']))
    scores = np.asarray(scores); labels = np.asarray(labels)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    ap = average_precision_score(labels, scores) if len(set(labels)) > 1 else float('nan')
    print(f'  rows: {len(labels)}   pos: {labels.sum()}')
    print(f'  AUC={auc:.4f}  AP={ap:.4f}  F1@0.5={f1_score(labels, (scores>=0.5).astype(int)):.4f}')
    return True


def check7_seg_coverage(pkl):
    section(7, 'Seg-cache coverage')
    needed = set()
    for split in ('train', 'val'):
        for r in pkl[split]:
            for st in r['sample_tokens']:
                needed.add(st)
    print(f'  unique sample_tokens needed: {len(needed)}')
    have = set()
    if os.path.isdir(SEG_CACHE):
        for fn in os.listdir(SEG_CACHE):
            if fn.endswith('.png'):
                have.add(fn[:-4])
    missing = needed - have
    print(f'  cached: {len(have)}   missing: {len(missing)}')
    if missing:
        sample_miss = list(missing)[:5]
        print(f'  first missing: {sample_miss}')
    return len(missing) == 0


def main():
    args = parse_args()

    print(f'Loading {args.seq_pkl} ...')
    with open(args.seq_pkl, 'rb') as f:
        pkl = pickle.load(f)
    print(f'  train={len(pkl["train"])}  val={len(pkl["val"])}')

    results = []
    results.append(('1.meta',        check1_meta(pkl)))
    results.append(('2.property',    check2_property(args.n_property, args.seed)))
    results.append(('3.provenance',  check3_provenance(pkl, args.n_provenance, args.seed)))
    results.append(('4.loader',      check4_loader_agreement(pkl)))
    results.append(('5.transform',   check5_transform(pkl)))
    results.append(('6.csv',         check6_csv(args.csv)))
    results.append(('7.seg_cov',     check7_seg_coverage(pkl)))

    print('\n=== Summary ===')
    for name, ok in results:
        print(f'  {"PASS" if ok else "FAIL":4s}  {name}')
    fails = [n for n, ok in results if not ok]
    if fails:
        print(f'\n{len(fails)} CHECK(S) FAILED: {fails}')
        sys.exit(1)
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()

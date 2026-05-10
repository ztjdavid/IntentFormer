"""
Evaluate IntentFormer (k=3) checkpoints on the seq3 nuScenes val split.

Reports per-checkpoint:
  - ROC AUC, average precision (PR-AUC)
  - accuracy / precision / recall / F1 at threshold 0.5
  - threshold maximising F1 (over PR curve points), with P/R/F1 at that thr.

Score = output of the third head (traj_o), softmax index 1 (class=Crossing).

Optionally dumps per-sample CSV with the same columns as
results/preds_future_intent_val.csv so EfficientPIE comparisons are 1:1.
"""

import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np

# Quiet TF logs.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)  # noqa: E402

from data_gen import IntentFormerSeqGen, load_records  # noqa: E402
from model import build_intentformer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index',
                   default='/root/IntentFormer/nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl')
    p.add_argument('--split', default='val', choices=['train', 'val', 'smoke'])
    p.add_argument('--weights', default=None,
                   help='single .h5 checkpoint path')
    p.add_argument('--weights-dir', default=None,
                   help='directory of *.h5 checkpoints (sweep mode)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--csv-out', default=None)
    p.add_argument('--no-seg', action='store_true')
    return p.parse_args()


def metrics_at_threshold(scores, labels, thresh):
    preds = (scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    n = len(labels)
    acc = (tp + tn) / n
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return dict(threshold=thresh, accuracy=acc, precision=p, recall=r,
                f1=f1, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def run_inference(model, gen, n_records):
    """Iterate gen, return (scores, labels) of length n_records.
    Note: gen drops the tail (n // batch_size); this returns only batched records.
    """
    scores, labels = [], []
    n = len(gen)
    for i in range(n):
        x, y = gen[i]
        out = model.predict_on_batch(x)
        # out is list-of-3, each (B, 2). Take traj_o (third), class 1 prob.
        traj_probs = out[2][:, 1]
        scores.append(traj_probs)
        labels.append(y)
    if not scores:
        return np.array([]), np.array([])
    return (np.concatenate(scores), np.concatenate(labels))


def evaluate_checkpoint(weights_path, gen, records_used):
    model = build_intentformer()
    model.load_weights(weights_path)
    scores, labels = run_inference(model, gen, len(records_used))

    if len(set(labels.tolist())) > 1:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        pr, rc, thr = precision_recall_curve(labels, scores)
        f1s = 2 * pr * rc / np.where((pr + rc) > 0, (pr + rc), 1)
        # thr has len = len(pr)-1; ignore the trailing element of f1s.
        if len(thr):
            best_idx = int(np.argmax(f1s[:-1]))
            best_thr = float(thr[best_idx])
        else:
            best_thr = 0.5
    else:
        auc = ap = float('nan')
        best_thr = 0.5

    m_default = metrics_at_threshold(scores, labels, 0.5)
    m_best = metrics_at_threshold(scores, labels, best_thr)
    return dict(weights=os.path.basename(weights_path), auc=auc, ap=ap,
                default=m_default, best=m_best, best_thr=best_thr,
                scores=scores, labels=labels)


def print_row(rep, header=False):
    cols = ['weights', 'auc', 'ap',
            'acc@0.5', 'p@0.5', 'r@0.5', 'f1@0.5',
            'best_thr', 'f1@best', 'p@best', 'r@best']
    if header:
        print('  '.join(f'{c:>16s}' for c in cols))
        print('-' * (18 * len(cols)))
    print('  '.join([
        f'{rep["weights"]:>16s}',
        f'{rep["auc"]:>16.4f}',
        f'{rep["ap"]:>16.4f}',
        f'{rep["default"]["accuracy"]:>16.4f}',
        f'{rep["default"]["precision"]:>16.4f}',
        f'{rep["default"]["recall"]:>16.4f}',
        f'{rep["default"]["f1"]:>16.4f}',
        f'{rep["best_thr"]:>16.4f}',
        f'{rep["best"]["f1"]:>16.4f}',
        f'{rep["best"]["precision"]:>16.4f}',
        f'{rep["best"]["recall"]:>16.4f}',
    ]))


def dump_csv(records, scores, labels, default_thr, best_thr, csv_out):
    os.makedirs(os.path.dirname(csv_out) or '.', exist_ok=True)
    n = min(len(records), len(scores))
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene_name', 'frame_idx', 'instance_token',
                    'sample_token', 'csv_label', 'label',
                    'pred_default', 'pred_best_f1', 'score'])
        for i in range(n):
            rec = records[i]
            sc = float(scores[i])
            lb = int(labels[i])
            w.writerow([rec['scene_name'], rec['frame_idx'],
                        rec['instance_token'], rec['sample_token'],
                        rec['csv_label'], lb,
                        int(sc >= default_thr), int(sc >= best_thr),
                        f'{sc:.4f}'])


def main():
    args = parse_args()

    if args.device.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.device.split(':')[1])

    print(f'Loading index from {args.index}')
    records, meta = load_records(args.index, args.split)
    n_pos = sum(r['label'] == 1 for r in records)
    print(f'  {args.split}: {len(records)} records '
          f'({n_pos} pos, {len(records) - n_pos} neg)')

    gen = IntentFormerSeqGen(
        records, batch_size=args.batch_size, k=meta.get('k', 3),
        shuffle=False, train=False, use_seg=not args.no_seg, seed=0)
    n_used = len(gen) * args.batch_size
    print(f'  using first {n_used} records (drop last partial batch)')
    records_used = records[:n_used]

    if args.weights_dir is not None:
        ckpts = sorted(glob.glob(os.path.join(args.weights_dir, 'cp_*.h5')))
        if not ckpts:
            print(f'ERROR: no cp_*.h5 in {args.weights_dir}', file=sys.stderr)
            sys.exit(2)
        print(f'Sweeping {len(ckpts)} checkpoints')
        results = []
        for i, ck in enumerate(ckpts):
            rep = evaluate_checkpoint(ck, gen, records_used)
            print_row(rep, header=(i == 0))
            results.append(rep)
        by_auc = sorted(results, key=lambda r: -r['auc'])[0]
        by_f1 = sorted(results, key=lambda r: -r['best']['f1'])[0]
        print('\nBest by AUC:    ', by_auc['weights'], f'AUC={by_auc["auc"]:.4f}')
        print('Best by best-F1:', by_f1['weights'], f'F1={by_f1["best"]["f1"]:.4f}')
    else:
        if args.weights is None:
            print('ERROR: pass --weights or --weights-dir', file=sys.stderr)
            sys.exit(2)
        rep = evaluate_checkpoint(args.weights, gen, records_used)
        print_row(rep, header=True)
        if args.csv_out:
            dump_csv(records_used, rep['scores'], rep['labels'],
                     0.5, rep['best_thr'], args.csv_out)
            print(f'Per-sample predictions -> {args.csv_out}')


if __name__ == '__main__':
    main()

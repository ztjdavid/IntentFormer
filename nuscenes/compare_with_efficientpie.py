"""
Apples-to-apples comparison of IntentFormer vs. EfficientPIE-vfuture_intent
on the intersection of their val record sets (matched on
(instance_token, sample_token)).

Reports AUC / AP / Acc / P / R / F1 for both models on the intersected subset,
plus the unintersected baselines for context.
"""

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--intentformer-csv',
                   default='/root/IntentFormer/nuscenes/results/intentformer_preds_vfuture_intent_val.csv')
    p.add_argument('--efficientpie-csv',
                   default='/root/EfficientPIE/results/preds_future_intent_val.csv')
    return p.parse_args()


def load(path):
    """Return dict {(instance_token, sample_token): {'label': int, 'score': float}}."""
    d = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row['instance_token'], row['sample_token'])
            d[key] = dict(label=int(row['label']),
                          score=float(row['score']),
                          scene_name=row.get('scene_name', ''),
                          frame_idx=int(row['frame_idx']),
                          csv_label=row.get('csv_label', ''))
    return d


def metrics(scores, labels, header):
    if len(set(labels.tolist())) <= 1:
        print(f'  {header}: only one class present -> skipping metrics')
        return
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    pr, rc, thr = precision_recall_curve(labels, scores)
    f1s = 2 * pr * rc / np.where((pr + rc) > 0, (pr + rc), 1)
    if len(thr):
        best_idx = int(np.argmax(f1s[:-1]))
        best_thr = float(thr[best_idx])
    else:
        best_thr = 0.5
    pred05 = (scores >= 0.5).astype(int)
    predB = (scores >= best_thr).astype(int)
    print(f'  {header}: '
          f'AUC={auc:.4f}  AP={ap:.4f}  '
          f'Acc@0.5={accuracy_score(labels, pred05):.4f}  '
          f'F1@0.5={f1_score(labels, pred05):.4f}  '
          f'F1@best({best_thr:.2f})={f1_score(labels, predB):.4f}  '
          f'P@best={precision_score(labels, predB, zero_division=0):.4f}  '
          f'R@best={recall_score(labels, predB, zero_division=0):.4f}')


def main():
    args = parse_args()
    if not os.path.exists(args.intentformer_csv):
        print(f'ERROR: {args.intentformer_csv} not found', file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.efficientpie_csv):
        print(f'ERROR: {args.efficientpie_csv} not found', file=sys.stderr)
        sys.exit(2)

    A = load(args.intentformer_csv)
    B = load(args.efficientpie_csv)
    print(f'IntentFormer rows: {len(A)}')
    print(f'EfficientPIE rows: {len(B)}')

    common = sorted(set(A.keys()) & set(B.keys()))
    print(f'Intersection on (instance_token, sample_token): {len(common)} records')

    # Sanity: labels should agree on intersection (same upstream JSON + intent_label).
    n_label_diff = sum(1 for k in common if A[k]['label'] != B[k]['label'])
    print(f'  label disagreements: {n_label_diff} / {len(common)}')

    if not common:
        return

    sa = np.array([A[k]['score'] for k in common])
    sb = np.array([B[k]['score'] for k in common])
    la = np.array([A[k]['label'] for k in common])
    lb = np.array([B[k]['label'] for k in common])

    print('\nOn the intersection:')
    metrics(sa, la, 'IntentFormer')
    metrics(sb, lb, 'EfficientPIE')

    print('\nFull (unintersected) baselines:')
    sa_full = np.array([v['score'] for v in A.values()])
    la_full = np.array([v['label'] for v in A.values()])
    sb_full = np.array([v['score'] for v in B.values()])
    lb_full = np.array([v['label'] for v in B.values()])
    metrics(sa_full, la_full, f'IntentFormer (n={len(la_full)})')
    metrics(sb_full, lb_full, f'EfficientPIE (n={len(lb_full)})')


if __name__ == '__main__':
    main()

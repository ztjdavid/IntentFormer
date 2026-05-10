"""
Train IntentFormer (k=3) on the nuScenes seq3 future-intent index.

Mirrors /root/EfficientPIE/train_EfficientPIE_nuscenes.py CLI surface.
"""

import argparse
import os
import sys
import time

# Make the package directory importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_gen import IntentFormerSeqGen, load_records
from model import build_intentformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index',
                   default='/root/IntentFormer/nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--version', default='vfuture_intent')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--train_limit', type=int, default=0,
                   help='if > 0, use first N (shuffled) train records (smoke).')
    p.add_argument('--val_limit', type=int, default=0,
                   help='if > 0, use first N (shuffled) val records (smoke).')
    p.add_argument('--shuffle', action='store_true',
                   help='shuffle records before --*_limit slicing.')
    p.add_argument('--no-seg', action='store_true',
                   help='zero out the seg input (ablation).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--weights-out',
                   default='/root/IntentFormer/nuscenes/weights_v_nuscenes')
    return p.parse_args()


def slice_records(records, limit, shuffle, seed):
    if limit and limit > 0:
        idx = np.arange(len(records))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        records = [records[i] for i in idx[:limit]]
    return records


def main():
    args = parse_args()

    if args.device.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.device.split(':')[1])
    keras.utils.set_random_seed(args.seed)

    out_dir = os.path.join(args.weights_out, f'v{args.version}')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train.log')

    # Tee stdout to train.log AND console.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()
    log_f = open(log_path, 'a', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    print(f'==== train.py @ {time.strftime("%Y-%m-%d %H:%M:%S")} ====')
    print('args:', vars(args))

    print(f'Loading seq index {args.index} ...')
    train_recs, meta = load_records(args.index, 'train')
    val_recs, _ = load_records(args.index, 'val')
    print(f'  train: {len(train_recs)}  val: {len(val_recs)}')
    print(f'  meta: {meta}')

    train_recs = slice_records(train_recs, args.train_limit, args.shuffle, args.seed)
    val_recs = slice_records(val_recs, args.val_limit, args.shuffle, args.seed + 1)
    print(f'after limits: train={len(train_recs)} val={len(val_recs)}')

    train_pos = sum(r['label'] for r in train_recs)
    val_pos = sum(r['label'] for r in val_recs)
    print(f'  train positives: {train_pos}/{len(train_recs)} '
          f'({100*train_pos/max(len(train_recs),1):.1f}%)')
    print(f'  val   positives: {val_pos}/{len(val_recs)} '
          f'({100*val_pos/max(len(val_recs),1):.1f}%)')

    train_gen = IntentFormerSeqGen(
        train_recs, batch_size=args.batch_size, k=meta.get('k', 3),
        shuffle=True, train=True, use_seg=not args.no_seg, seed=args.seed)
    val_gen = IntentFormerSeqGen(
        val_recs, batch_size=args.batch_size, k=meta.get('k', 3),
        shuffle=False, train=False, use_seg=not args.no_seg, seed=args.seed + 2)
    print(f'steps/epoch: train={len(train_gen)} val={len(val_gen)}')

    print('Building model ...')
    model = build_intentformer()
    print(f'params: {model.count_params():,}')

    optimizer = keras.optimizers.Adam(
        learning_rate=args.lr, weight_decay=args.weight_decay
    ) if hasattr(keras.optimizers.Adam, 'weight_decay') else keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )

    # Cell-27 callbacks.
    def scheduler(epoch, lr):
        if epoch < 7:
            return lr
        return lr * tf.math.exp(-0.1)

    sched = keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5)
    early = keras.callbacks.EarlyStopping(
        monitor='val_traj_o_accuracy', min_delta=0, patience=7, mode='max',
        restore_best_weights=True, verbose=1)
    cp = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(out_dir, 'cp_{epoch:02d}.h5'),
        save_weights_only=True, save_freq='epoch', verbose=1,
        monitor='val_traj_o_accuracy', mode='max')
    csv_logger = keras.callbacks.CSVLogger(
        os.path.join(out_dir, 'history.csv'), append=True)

    print('Starting fit ...')
    t0 = time.time()
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=args.epochs,
              callbacks=[reduce_lr, sched, cp, early, csv_logger],
              workers=4, max_queue_size=8, use_multiprocessing=False,
              verbose=2)
    elapsed = time.time() - t0
    print(f'Done. Total fit wall-time: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()

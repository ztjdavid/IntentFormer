# IntentFormer × nuScenes — `vfuture_intent` (k=3)

## What this is
[IntentFormer](https://www.sciencedirect.com/science/article/pii/S0950705124005033)
([paper.md](/root/IntentFormer/paper.md)) trained on the same 321/150-scene
nuScenes pedestrian-intent setup that
[/root/EfficientPIE/CLAUDE.md](/root/EfficientPIE/CLAUDE.md) (the
`vfuture_intent` run) uses, with one tweak: **k=3** observation length
(matches UniAD's `queue_length=3` for stage-2 intent in
[base_intent_fornt.py](/root/UniAD/projects/configs/stage2_e2e/base_intent_fornt.py)),
vs. the paper's k=14 designed for 30 fps PIE/JAAD (nuScenes keyframes are 2 Hz).

Goal: an apples-to-apples IntentFormer baseline against
EfficientPIE's `vfuture_intent` (AUC 0.6944, F1@best 0.4176).

## Design table

| Decision | Choice |
|---|---|
| Sampling unit | One sample per `(instance_token, anchor_keyframe)` with a 3-frame visible window `[t-2, t-1, t]`. |
| Label mapping | binary; from `intent_label(look_ahead=4)` (verbatim from UniAD `nuscenes_e2e_dataset.py:41-92`): intent7==2 → 1; intent7∈{0,1} → 0; intent7==−1 dropped. |
| Window size (k) | 3 keyframes (anchor + 2 prior). Drop records where instance not visible in all 3. |
| Camera | CAM_FRONT only. |
| Train scenes | 321 from `/mnt/storage/EfficientPIE/nuscenes_infos_temporal_train.pkl`. |
| Val scenes | 150 official via `nuscenes.utils.splits.create_splits_scenes()`. |
| Filter | category `human.pedestrian.*`, `num_lidar_pts ≥ 1` OR `num_radar_pts ≥ 1`, 2D bbox side ≥ 20 px, in CAM_FRONT — **applied to every frame in the window**. |
| Crop | `squarify` + `img_pad('pad_resize', 224)` (notebook cells 6,7,9). |
| Segmentation | precomputed once with SegFormer-B0 (ADE20k) on each unique CAM_FRONT keyframe; cached as palette PNG at `/mnt/storage/IntentFormer/seg_cache/<sample_token>.png`. |
| Class balancing | none (matches `vfuture_intent`); `--no-seg` ablates the seg modality. |
| Architecture | IntentFormer with `INPUT_SHAPE=(3,224,224,3)`, `PATCH_SIZE=(1,8,8)` (preserves k=3 temporal info; was (2,8,8) at k=14). 5.8 M params. |
| Loss | sum of `SparseCategoricalCrossentropy` over 3 heads (rgb_o, seg_o, traj_o), matching the published `model.compile()`. |
| Optimizer | Adam(lr=1e-4) + ReduceLROnPlateau + exp decay after epoch 7; EarlyStopping(monitor='val_traj_o_accuracy', patience=7, restore_best). |
| Headline score | `traj_o[:,1]` (3rd head, full RGB+seg+traj representation). |

## Data layout

| What | Path |
|---|---|
| nuScenes raw images | `/mnt/nuscenes/nuScenes/samples/CAM_FRONT/*.jpg` |
| nuScenes JSON labels | `/mnt/nuscenes/nuScenes/unified_map_v3/all_scenes_compact_new.json` |
| Train scene set | `/mnt/storage/EfficientPIE/nuscenes_infos_temporal_train.pkl` |
| Built seq3 index | `data/nuscenes_ped_intent_seq3_v2.pkl` |
| SegFormer cache | `/mnt/storage/IntentFormer/seg_cache/<sample_token>.png` |

## Index stats (`data/nuscenes_ped_intent_seq3_v2.pkl`)

```
train: 13 433 records   label_dist={1: 3 656 (27.2%), 0: 9 777 (72.8%)}
val:    3 660 records   label_dist={1:   793 (21.7%), 0: 2 867 (78.3%)}
intent_7class (train):  STOPPED=3098  MOVING=6679  Crossing=3656
intent_7class (val):    STOPPED=1005  MOVING=1862  Crossing=793
drop reasons:
  intent_undefined        307 519
  not_a_pedestrian         76 235
  insufficient_history     22 842   (frame_idx < k-1 = 2)
  no_annotation_in_window  12 388
  non_pedestrian_intent     8 150
  no_sensor_return_in_window 7 301
  bbox_too_small_in_window  3 371
  not_in_cam_front_in_window  537
```

EfficientPIE-vfuture_intent (per-frame, single image) has 19 251 train / 5 682 val.
Our seq3 filter (require k=3 frames visible) drops ~30% (train) and ~36% (val).

## Files

| File | Purpose |
|---|---|
| [build_nuscenes_seq_index_v2.py](build_nuscenes_seq_index_v2.py) | Builds the seq3 pkl. Embeds `intent_label()` verbatim. |
| [precompute_segformer.py](precompute_segformer.py) | One-time SegFormer cache (resumable). |
| [model.py](model.py) | Extract of notebook cells 21-25. `build_intentformer(input_shape=(3,224,224,3), patch_size=(1,8,8))`. |
| [data_gen.py](data_gen.py) | TF Keras `Sequence` (fixes notebook's `np.random.randint` shuffle bug). |
| [train.py](train.py) | Train loop with cell-27 callbacks; saves H5 weights `cp_{epoch:02d}.h5`. |
| [eval.py](eval.py) | Sweep `cp_*.h5`, AUC/AP/F1, per-sample CSV. |
| [viz_seq.py](viz_seq.py) | Per-instance temporal strips: full anchor + RGB strip + seg strip. |
| [verify.py](verify.py) | 7-check verification suite. |
| [smoke.sh](smoke.sh) | 1-epoch, 200/200 records smoke wrapper. |

## Verification (7 checks)

| # | Check | Result |
|---|---|---|
| 1 | Index meta (`k=3`, `look_ahead=4`, `seg_cache_dir`, `source_pkl`, `source_json`) | PASS |
| 2 | `intent_label()` property test (134 953 random calls vs EfficientPIE source) | PASS — 0 mismatches |
| 3 | End-to-end label provenance (16 random pkl records vs raw JSON) | PASS — 0 mismatches |
| 4 | Train/eval loader agreement (record counts) | PASS |
| 5 | Transform agreement (val/train deterministic-pipeline equality, no aug) | PASS |
| 6 | Metrics-from-CSV reproducibility | _filled in after eval CSV is dumped_ |
| 7 | Seg-cache coverage (every needed `sample_token` has a cached PNG) | _filled in after precompute completes_ |

Run: `python3 verify.py [--csv results/intentformer_preds_vfuture_intent_val.csv]`.

## Environment

- 1× NVIDIA RTX 3090, 24 GB.
- `tensorflow==2.10.1`, `transformers==4.30.2`, `nvidia-cudnn-cu11==8.6.0.163`,
  `nvidia-cublas-cu11==11.10.3.66`, `torch==1.9.1+cu111` (used by SegFormer).
- `LD_LIBRARY_PATH` must include `nvidia/cudnn/lib` and `nvidia/cublas/lib`
  (`smoke.sh` and `train.py` set this; persisted in `/root/.bashrc`).

## Commands

```bash
# 1. (one-time) build seq3 index
python3 nuscenes/build_nuscenes_seq_index_v2.py

# 2. (one-time, ~25 min on 3090) precompute SegFormer
python3 nuscenes/precompute_segformer.py --batch-size 16

# 3. verify
python3 nuscenes/verify.py

# 4. smoke train
bash nuscenes/smoke.sh

# 5. visualisation
python3 nuscenes/viz_seq.py \
    --index nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl \
    --split train --out-dir nuscenes/viz_seq_v2/full \
    --num-instances 10 --per-instance 3 --stratified

# 6. full train
python3 nuscenes/train.py \
    --index nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl \
    --epochs 50 --batch_size 8 --lr 1e-4 \
    --version vfuture_intent

# 7. eval sweep + headline csv
python3 nuscenes/eval.py \
    --index nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl \
    --weights-dir nuscenes/weights_v_nuscenes/vvfuture_intent/ --split val
python3 nuscenes/eval.py \
    --index nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl \
    --weights nuscenes/weights_v_nuscenes/vvfuture_intent/cp_<best>.h5 \
    --csv-out nuscenes/results/intentformer_preds_vfuture_intent_val.csv
```

## Headline metrics

_TBD — filled in after the full training and eval sweep completes._

## Comparison vs. EfficientPIE-vfuture_intent

_TBD — intersect on `(instance_token, sample_token)`._

## Open questions

- **Seg domain shift**: SegFormer/ADE20k → swap to Cityscapes if smoke viz looks bad.
- **Notebook artefacts not carried over**: bbox jitter (commented out in cell 12),
  per-modality independent random shuffle (was a bug; we use single-list permutation),
  `Custom_CE_Loss` learnable weights (defined cell 24 but unused by `model.compile()`).

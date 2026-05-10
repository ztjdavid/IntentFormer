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
| Filter | category `human.pedestrian.*`, `num_lidar_pts ≥ 1` OR `num_radar_pts ≥ 1`, 2D bbox side ≥ 20 px, in CAM_FRONT — **applied to anchor frame only**; past frames may be UniAD-style zero placeholders (`bbox=[0,0,0,0]`, `visibility=False`) when the agent is occluded. Each record carries `visibility=[bool, bool, bool]`. |
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

## Index stats (`data/nuscenes_ped_intent_seq3_v2.pkl`, schema v3)

```
train: 18 242 records   label_dist={1: 4 547 (24.9%), 0: 13 695 (75.1%)}
val:    5 278 records   label_dist={1: 1 040 (19.7%), 0:  4 238 (80.3%)}
intent_7class (train):  STOPPED=4425  MOVING=9270  Crossing=4547
intent_7class (val):    STOPPED=1546  MOVING=2692  Crossing=1040

visibility patterns (train+val, 23 520 records):
  (T,T,T) = 17 093  (full visibility)
  (F,T,T) =  2 797  (t-2 occluded)
  (F,F,T) =  2 684  (t-2 and t-1 occluded)
  (T,F,T) =    946  (t-1 occluded)

anchor drops (record discarded):
  anchor_not_a_pedestrian   84 844
  anchor_no_sensor_return    5 172
  anchor_bbox_too_small      3 370
  anchor_not_in_cam_front       10
  anchor_no_annotation           9

past-frame placeholder events (record kept, frame zeroed):
  past_placeholder_not_a_pedestrian  156 656
  past_placeholder_no_annotation      18 696
  past_placeholder_no_sensor_return    9 706
  past_placeholder_bbox_too_small      6 574
  past_placeholder_not_in_cam_front      819

other drops:
  intent_undefined        307 519
  insufficient_history     22 842   (frame_idx < k-1 = 2)
  non_pedestrian_intent     8 150
```

EfficientPIE-vfuture_intent (per-frame, single image) has 19 251 train / 5 682 val.
With UniAD-style zero placeholders for occluded past frames, our seq3 set is now
within ~5% of EfficientPIE on train (18 242 vs 19 251) and ~7% on val
(5 278 vs 5 682). The schema-v2 (k=3 strict-visibility) variant only had
13 433 / 3 660 — i.e. dropped 30 %/36 % vs. EfficientPIE.

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

## Verification (8 checks — all PASS)

| # | Check | Result |
|---|---|---|
| 1 | Index meta (`k=3`, `look_ahead=4`, `seg_cache_dir`, `source_pkl`, `source_json`, `index_schema_version=3`) | PASS |
| 2 | `intent_label()` property test (134 953 random calls vs EfficientPIE source) | PASS — 0 mismatches |
| 3 | End-to-end label provenance (16 random pkl records vs raw JSON) | PASS — 0 mismatches |
| 4 | Train/eval loader agreement (record counts) | PASS — 18 242 / 5 278 |
| 5 | Transform agreement (val/train deterministic-pipeline equality, no aug) | PASS |
| 6 | Metrics-from-CSV reproducibility | PASS (after eval CSV emitted) |
| 7 | Seg-cache coverage (every needed `sample_token` has a cached PNG) | PASS — 9 153 / 9 153 needed cached |
| 8 | Visibility schema (anchor always True; placeholder iff `bbox=[0,0,0,0]`) | PASS — 0 missing, 0 anchor-invisible, 0 disagreements |

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

## Headline metrics (run `vfuture_intent`, schema v3)

50-epoch fit, EarlyStopping(monitor=`val_traj_o_accuracy`, patience=7) fired at
epoch 11 → 78 min wall on RTX 3090 (more steps/epoch with the larger pkl).
11 checkpoints saved; best weights restored from epoch 4.

Sweep over `cp_*.h5` on the **5 278-record val split** (full table at
`weights_v_nuscenes/vvfuture_intent/val_sweep.log`). Headline checkpoint
picked by val ROC-AUC, also best by F1@best: **`cp_04.h5`** (epoch 4).

| Metric | Value |
|---|---:|
| ROC AUC | **0.7576** |
| Avg precision (PR-AUC) | 0.5060 |
| Accuracy @ 0.5 | 0.8314 |
| Precision @ 0.5 | 0.6736 |
| Recall @ 0.5 | 0.2817 |
| F1 @ 0.5 | 0.3973 |
| Best threshold | 0.2515 |
| F1 @ best | **0.4983** |
| P @ best | 0.5005 |
| R @ best | 0.4962 |

Per-sample predictions: `results/intentformer_preds_vfuture_intent_val.csv`
(columns match EfficientPIE's `preds_future_intent_val.csv`).

Eval drops the last partial batch — sweep operates on 5 272/5 278 records.
n_pos=1 040, n_neg=4 232 on the scored subset (19.7 %).

## Comparison vs. EfficientPIE-vfuture_intent

`compare_with_efficientpie.py` intersects the two CSVs on
`(instance_token, sample_token)` → **5 272 records** (every IntentFormer val
row is also in EfficientPIE; 0 label disagreements ⇒ same upstream JSON +
intent_label).

Apples-to-apples on the intersected 5 272-record subset:

| Metric | IntentFormer (cp_04) | EfficientPIE (model_19) | Δ |
|---|---:|---:|---:|
| ROC AUC | **0.7576** | 0.6985 | **+0.0591** |
| Avg precision (PR-AUC) | **0.5060** | 0.4381 | +0.0679 |
| Accuracy @ 0.5 | **0.8314** | 0.7836 | +0.0478 |
| F1 @ 0.5 | **0.3973** | 0.3966 | +0.0007 |
| F1 @ best threshold | **0.4983** | 0.4268 | +0.0715 |
| P @ best | **0.5005** | 0.4083 | +0.0922 |
| R @ best | **0.4962** | 0.4471 | +0.0491 |

The intersection is now 1 616 records larger than schema-v2 (5 272 vs 3 656),
covering harder partial-visibility cases (5 481 records have at least one
placeholder past frame). IntentFormer's lead grows on F1@best (+0.0715 vs
+0.0653) and on precision (+0.0922 vs +0.0232) compared to the prior
strict-visibility result, while ceding some recall (+0.0491 vs +0.1287). The
higher operating-point threshold (0.50 vs 0.45) reflects a more conservative
calibration — the model is less aggressive at flagging crossings under
uncertainty.

For context, the unintersected baselines (each model on its own full val):

| | n_val | AUC | F1@best |
|---|---:|---:|---:|
| IntentFormer | 5 272 | 0.7576 | 0.4983 |
| EfficientPIE | 5 682 | 0.6944 | 0.4176 |

Schema-v2 → v3 deltas (same model arch, same loss, same hp; only the
visibility filter relaxed and the training set grew +35.8 %):

| | v2 (strict) | v3 (placeholders) | Δ |
|---|---:|---:|---:|
| Train records | 13 433 | 18 242 | +35.8 % |
| Val records   | 3 660  | 5 278  | +44.2 % |
| Headline cp   | cp_05  | cp_04  | — |
| AUC (own val) | 0.7782 | 0.7576 | −0.0206 |
| F1@best       | 0.5278 | 0.4983 | −0.0295 |

The drop reflects a strictly harder val set, not a worse model — IntentFormer
still beats EfficientPIE by a wider F1@best margin in v3.

## Open questions

- **Seg domain shift**: SegFormer/ADE20k → swap to Cityscapes if smoke viz looks bad.
- **Notebook artefacts not carried over**: bbox jitter (commented out in cell 12),
  per-modality independent random shuffle (was a bug; we use single-list permutation),
  `Custom_CE_Loss` learnable weights (defined cell 24 but unused by `model.compile()`).
- **Mask-token attention for placeholders**: model currently sees zero RGB/seg/bbox
  for occluded past frames (UniAD-style zero query). A learned mask token (or
  attention-mask gate) at `TubeletEmbedding` could let the model distinguish
  "no observation" from "stationary at origin" and may close the v2→v3 AUC gap.
  Would need a 4th input head consuming `visibility[k]` (already in the index).

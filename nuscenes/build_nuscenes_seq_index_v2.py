"""
Build a sequence-aware (k=3) index for IntentFormer-on-nuScenes.

For each (pedestrian instance, anchor keyframe) where:
  - intent_label() at the anchor returns >= 0 (look_ahead = 4),
  - the same instance is visible (3D-projected, sensor-return, bbox >= --min-bbox)
    in EVERY frame of the window [anchor - (k-1) .. anchor],
  - the JSON's per-instance action_array spans the full window,
  - category_name starts with 'human.pedestrian.',
write a record:
    {img_paths[k], seg_paths[k], bboxes[k] (each [x1,y1,x2,y2]),
     ego_speeds[k] (m/s), sample_tokens[k],
     label (0/1), csv_label, intent_7class,
     instance_token, sample_token (anchor), scene_token, scene_name, frame_idx}

Binary label mapping (same as EfficientPIE vfuture_intent):
  Crossing (intent7 == 2)              -> 1
  STOPPED / MOVING (intent7 in {0,1})   -> 0
  intent_label() == -1                  -> dropped

Train scene set: 321 scene_tokens from
  /mnt/storage/EfficientPIE/nuscenes_infos_temporal_train.pkl
Val scene set: 150 official nuScenes val scenes via create_splits_scenes().

intent_label() copied verbatim from
/root/UniAD/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py:41-92
(via /root/EfficientPIE/utils/build_nuscenes_index_v2.py:47-95).
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.utils.splits import create_splits_scenes


# Copied verbatim from
# /root/UniAD/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py:41-92
def intent_label(index, action_array):
    intent_dic = {
        "Crossing": 2,
        "TURN_RIGHT": 3,
        "TURN_LEFT": 4,
        "LANE_CHANGE_RIGHT": 5,
        "LANE_CHANGE_LEFT": 6,
        "STOPPED": 0,
        "MOVING": 1,
        "Stopped": 0,
        "Moving": 1,
    }
    if index < 0 or index >= len(action_array) - 1:
        return -1

    action = action_array[index]

    end = min(len(action_array), index + 5)
    future_actions = action_array[index + 1:end]
    if len(future_actions) == 0 or all(a == "na" for a in future_actions):
        return -1

    action_set = {"Crossing", "TURN_RIGHT", "TURN_LEFT",
                  "LANE_CHANGE_RIGHT", "LANE_CHANGE_LEFT"}

    if action in action_set:
        next_action = future_actions[0]
        if next_action != action and next_action != "na":
            return intent_dic[next_action]
        else:
            return intent_dic[action]
    elif action == "STOPPED" or action == "Stopped":
        next_moving = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action]
            elif next_action == "MOVING" or next_action == "Moving":
                next_moving = True
        return intent_dic["MOVING"] if next_moving else intent_dic["STOPPED"]
    elif action == "MOVING" or action == "Moving":
        next_stop = False
        for next_action in future_actions:
            if next_action in action_set:
                return intent_dic[next_action]
            elif next_action == "STOPPED" or next_action == "Stopped":
                next_stop = True
        return intent_dic["STOPPED"] if next_stop else intent_dic["MOVING"]
    else:
        return -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--nusc-root', default='/mnt/nuscenes/nuScenes')
    p.add_argument('--nusc-version', default='v1.0-trainval')
    p.add_argument('--train-pkl',
                   default='/mnt/storage/EfficientPIE/nuscenes_infos_temporal_train.pkl',
                   help='UniAD/mmdet3d-style infos pkl. Used only to enumerate '
                        'the filtered train scene_tokens (321 scenes).')
    p.add_argument('--json',
                   default='/mnt/nuscenes/nuScenes/unified_map_v3/all_scenes_compact_new.json',
                   help='Per-(scene, instance) action label arrays.')
    p.add_argument('--seg-cache-dir',
                   default='/mnt/storage/IntentFormer/seg_cache',
                   help='Where SegFormer cached PNGs live (one per sample_token).')
    p.add_argument('--out',
                   default='/root/IntentFormer/nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl')
    p.add_argument('--k', type=int, default=3,
                   help='temporal window size (anchor + k-1 prior frames)')
    p.add_argument('--min-bbox', type=int, default=20,
                   help='min 2D bbox side in pixels (per frame in window)')
    p.add_argument('--scene', default=None,
                   help='if set, only process this scene name (smoke run)')
    return p.parse_args()


def walk_samples(nusc, scene):
    samples = []
    tok = scene['first_sample_token']
    while tok:
        s = nusc.get('sample', tok)
        samples.append(s)
        tok = s['next']
    return samples


def project_to_cam_front(nusc, sample, ann_token, image_w=1600, image_h=900):
    """Return (img_path, [x1,y1,x2,y2]) or None if not visible."""
    cam_token = sample['data']['CAM_FRONT']
    data_path, boxes, intrinsic = nusc.get_sample_data(
        cam_token,
        box_vis_level=BoxVisibility.ANY,
        selected_anntokens=[ann_token],
    )
    if not boxes:
        return None
    box = boxes[0]
    corners_3d = box.corners()
    if np.any(corners_3d[2, :] <= 0.1):
        return None
    pts2d = view_points(corners_3d, np.asarray(intrinsic), normalize=True)[:2]
    x1, y1 = pts2d.min(axis=1)
    x2, y2 = pts2d.max(axis=1)
    x1c = max(0.0, x1)
    y1c = max(0.0, y1)
    x2c = min(float(image_w - 1), x2)
    y2c = min(float(image_h - 1), y2)
    if x2c - x1c <= 1 or y2c - y1c <= 1:
        return None
    return data_path, [float(x1c), float(y1c), float(x2c), float(y2c)]


def get_ego_pose_for_cam_front(nusc, sample):
    sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    return nusc.get('ego_pose', sd['ego_pose_token'])


def compute_ego_speeds(nusc, samples):
    """Per-keyframe ego speed in m/s, using CAM_FRONT ego_pose finite differences.
    speed[i] = || pos[i] - pos[i-1] || / dt[i, i-1]   (forward-diff at i==0)
    """
    n = len(samples)
    if n == 0:
        return []
    poses = [get_ego_pose_for_cam_front(nusc, s) for s in samples]
    pos = [np.asarray(p['translation']) for p in poses]
    ts = [p['timestamp'] for p in poses]   # microseconds
    speeds = [0.0] * n
    for i in range(n):
        if i == 0:
            if n < 2:
                speeds[i] = 0.0
                continue
            j, k = 0, 1
        else:
            j, k = i - 1, i
        dt = (ts[k] - ts[j]) / 1e6
        if dt <= 1e-6:
            speeds[i] = 0.0
        else:
            speeds[i] = float(np.linalg.norm(pos[k] - pos[j]) / dt)
    return speeds


def build_seq_records_for_scene(nusc, scene, json_data, k, min_bbox, seg_cache_dir):
    """Return (records, drop_reasons Counter) for one scene."""
    if scene['token'] not in json_data:
        return [], Counter({'scene_not_in_json': 1})

    samples = walk_samples(nusc, scene)
    n_samples = len(samples)
    records = []
    drop_reasons = Counter()
    scene_data = json_data[scene['token']]

    sample_inst_to_ann = []
    for s in samples:
        m = {}
        for ann_tok in s['anns']:
            a = nusc.get('sample_annotation', ann_tok)
            m[a['instance_token']] = a
        sample_inst_to_ann.append(m)

    ego_speeds_all = compute_ego_speeds(nusc, samples)

    for inst_tok, agent_info in scene_data.items():
        action_array = agent_info['labels']
        n_frames = min(n_samples, len(action_array))
        for frame_idx in range(n_frames):
            # Window = [frame_idx-(k-1), ..., frame_idx]
            if frame_idx < k - 1:
                drop_reasons['insufficient_history'] += 1
                continue

            intent7 = intent_label(frame_idx, action_array)
            if intent7 == -1:
                drop_reasons['intent_undefined'] += 1
                continue
            if intent7 == 2:
                label = 1
            elif intent7 in (0, 1):
                label = 0
            else:
                drop_reasons['non_pedestrian_intent'] += 1
                continue

            window_ok = True
            window_frames = []
            for fidx in range(frame_idx - (k - 1), frame_idx + 1):
                sample = samples[fidx]
                ann = sample_inst_to_ann[fidx].get(inst_tok)
                if ann is None:
                    drop_reasons['no_annotation_in_window'] += 1
                    window_ok = False
                    break
                if not ann['category_name'].startswith('human.pedestrian.'):
                    drop_reasons['not_a_pedestrian'] += 1
                    window_ok = False
                    break
                if ann['num_lidar_pts'] == 0 and ann['num_radar_pts'] == 0:
                    drop_reasons['no_sensor_return_in_window'] += 1
                    window_ok = False
                    break
                proj = project_to_cam_front(nusc, sample, ann['token'])
                if proj is None:
                    drop_reasons['not_in_cam_front_in_window'] += 1
                    window_ok = False
                    break
                img_path, bbox = proj
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < min_bbox or h < min_bbox:
                    drop_reasons['bbox_too_small_in_window'] += 1
                    window_ok = False
                    break
                window_frames.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'sample_token': sample['token'],
                    'ego_speed': ego_speeds_all[fidx],
                    'seg_path': os.path.join(seg_cache_dir, f"{sample['token']}.png"),
                })
            if not window_ok:
                continue

            records.append({
                'img_paths': [f['img_path'] for f in window_frames],
                'seg_paths': [f['seg_path'] for f in window_frames],
                'bboxes': [f['bbox'] for f in window_frames],
                'ego_speeds': [f['ego_speed'] for f in window_frames],
                'sample_tokens': [f['sample_token'] for f in window_frames],
                'label': label,
                'csv_label': action_array[frame_idx],
                'instance_token': inst_tok,
                'sample_token': samples[frame_idx]['token'],
                'scene_token': scene['token'],
                'scene_name': scene['name'],
                'frame_idx': frame_idx,
                'intent_7class': intent7,
            })
    return records, drop_reasons


def main():
    args = parse_args()
    print(f'Loading nuScenes ({args.nusc_version}) from {args.nusc_root} ...')
    nusc = NuScenes(version=args.nusc_version,
                    dataroot=args.nusc_root, verbose=False)
    print(f'  {len(nusc.scene)} scenes loaded')

    print(f'Reading train scene_tokens from {args.train_pkl} ...')
    with open(args.train_pkl, 'rb') as f:
        pkl = pickle.load(f)
    train_scene_tokens = {info['scene_token'] for info in pkl['infos']}
    print(f'  {len(train_scene_tokens)} unique scene_tokens in train pkl')

    splits = create_splits_scenes()
    val_names = set(splits['val'])
    print(f'  official val: {len(val_names)} scenes')

    print(f'Loading intent JSON from {args.json} ...')
    with open(args.json) as f:
        json_data = json.load(f)
    print(f'  {len(json_data)} scenes in JSON')

    if args.scene:
        target_scene = next((s for s in nusc.scene if s['name'] == args.scene), None)
        if target_scene is None:
            print(f'ERROR: scene {args.scene!r} not in nuScenes', file=sys.stderr)
            sys.exit(2)
        recs, drops = build_seq_records_for_scene(
            nusc, target_scene, json_data, args.k, args.min_bbox, args.seg_cache_dir)
        label_dist = Counter(r['label'] for r in recs)
        intent7_dist = Counter(r['intent_7class'] for r in recs)
        print(f'scene {args.scene}: {len(recs)} records '
              f'label_dist={dict(label_dist)} intent7={dict(intent7_dist)}')
        print(f'  drops={dict(drops)}')
        out = args.out.replace('.pkl', '.smoke.pkl')
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump({'smoke': recs,
                         'meta': {
                             'k': args.k,
                             'look_ahead': 4,
                             'drop_reasons': dict(drops),
                             'seg_cache_dir': args.seg_cache_dir,
                             'source_pkl': args.train_pkl,
                             'source_json': args.json,
                         }}, f)
        print(f'wrote smoke seq index -> {out}')
        return

    train_records, val_records = [], []
    total_drops = Counter()
    n_skip = 0
    t0 = time.time()
    n_total = len(nusc.scene)
    for i, scene in enumerate(nusc.scene):
        if scene['token'] in train_scene_tokens:
            target = train_records
        elif scene['name'] in val_names:
            target = val_records
        else:
            n_skip += 1
            continue
        recs, drops = build_seq_records_for_scene(
            nusc, scene, json_data, args.k, args.min_bbox, args.seg_cache_dir)
        target.extend(recs)
        total_drops.update(drops)
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            print(f'  {i + 1}/{n_total} scenes  '
                  f'train={len(train_records)}  val={len(val_records)}  '
                  f'({elapsed:.1f}s)')

    label_dist = lambda recs: Counter(r['label'] for r in recs)
    intent7_dist = lambda recs: Counter(r['intent_7class'] for r in recs)
    print(f'\nDone. Skipped {n_skip} scenes (not train pkl, not val).')
    print(f'  train: {len(train_records)} records, '
          f'label_dist={dict(label_dist(train_records))}, '
          f'intent7={dict(intent7_dist(train_records))}')
    print(f'  val:   {len(val_records)} records, '
          f'label_dist={dict(label_dist(val_records))}, '
          f'intent7={dict(intent7_dist(val_records))}')
    print(f'  drop reasons: {dict(total_drops)}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({
            'train': train_records,
            'val': val_records,
            'meta': {
                'k': args.k,
                'look_ahead': 4,
                'min_bbox': args.min_bbox,
                'nusc_version': args.nusc_version,
                'drop_reasons': dict(total_drops),
                'n_train_scenes': len(train_scene_tokens),
                'n_val_scenes': len(val_names),
                'source_pkl': args.train_pkl,
                'source_json': args.json,
                'seg_cache_dir': args.seg_cache_dir,
            },
        }, f)
    print(f'wrote -> {args.out}')


if __name__ == '__main__':
    main()

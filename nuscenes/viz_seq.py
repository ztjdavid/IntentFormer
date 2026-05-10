"""
Visualize sequence-aware (k=3) seq3 index records.

Outputs to <out-dir>/per_instance/<inst[:8]>_<frame>.png:
  - one PNG per stratified record
  - top: full CAM_FRONT anchor frame with bbox overlay and label text
  - middle row: k cropped RGB frames (left = oldest)
  - bottom row: k cropped seg frames (or grey placeholders if seg missing)

Usage:
  python3 viz_seq.py \
      --index data/nuscenes_ped_intent_seq3_v2.smoke.pkl \
      --split smoke --out-dir viz_seq_smoke \
      --num-instances 5 --stratified
"""

import argparse
import collections
import os
import pickle
import random
import sys

from PIL import Image, ImageDraw, ImageFont


CROSS_COLOR = (220, 30, 30)
NOT_CROSS_COLOR = (30, 180, 30)
THUMB = 224
PAD = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--split', default='auto')
    p.add_argument('--out-dir', default='viz_seq_v2')
    p.add_argument('--num-instances', type=int, default=10,
                   help='unique pedestrians to visualise')
    p.add_argument('--per-instance', type=int, default=3,
                   help='records per pedestrian')
    p.add_argument('--stratified', action='store_true',
                   help='evenly draw from label=0 and label=1')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def color_and_text(label, csv_label):
    if label == 1:
        return CROSS_COLOR, f'CROSS ({csv_label})'
    return NOT_CROSS_COLOR, f'NOT-CROSS ({csv_label})'


def squarify(box, img_w):
    x1, y1, x2, y2 = box
    w = abs(x1 - x2)
    h = abs(y1 - y2)
    delta = h - w
    x1 -= delta / 2
    x2 += delta / 2
    if x1 < 0:
        x1 = 0
    if x2 > img_w:
        x1 = x1 - (x2 - img_w)
        x2 = img_w
    return [max(0, x1), max(0, y1), min(img_w, x2), y2]


def crop_to_thumb(img, box, size=THUMB):
    box = squarify(list(box), img.size[0])
    box = [int(round(v)) for v in box]
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(img.size[0] - 1, box[2])
    box[3] = min(img.size[1] - 1, box[3])
    if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        return Image.new('RGB', (size, size), color=(64, 64, 64))
    crop = img.crop(box)
    ratio = size / max(crop.size)
    new_size = (int(crop.size[0] * ratio), int(crop.size[1] * ratio))
    crop = crop.resize(new_size, Image.NEAREST)
    canvas = Image.new('RGB', (size, size), color=(0, 0, 0))
    canvas.paste(crop, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def render_record(rec, font, big_font):
    """Compose full-frame + RGB strip + seg strip into a single PNG."""
    k = len(rec['img_paths'])
    color, txt = color_and_text(rec['label'], rec['csv_label'])

    # 1) full-frame anchor with bbox overlay.
    full = Image.open(rec['img_paths'][-1]).convert('RGB').copy()
    draw = ImageDraw.Draw(full)
    x1, y1, x2, y2 = rec['bboxes'][-1]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    header = (f'{txt}  inst={rec["instance_token"][:8]}  '
              f'frame={rec["frame_idx"]:02d}  '
              f'speed={rec["ego_speeds"][-1]:.1f}m/s  '
              f'scene={rec["scene_name"]}')
    tw = draw.textlength(header, font=font)
    draw.rectangle([0, 0, tw + 16, 28], fill=color)
    draw.text((8, 4), header, fill='white', font=font)

    full_w = 800
    full_h = int(full.size[1] * full_w / full.size[0])
    full = full.resize((full_w, full_h), Image.BILINEAR)

    # 2) RGB strip and seg strip.
    rgb_thumbs = []
    seg_thumbs = []
    for ip, sp, bb in zip(rec['img_paths'], rec['seg_paths'], rec['bboxes']):
        rgb_img = Image.open(ip).convert('RGB')
        rgb_thumbs.append(crop_to_thumb(rgb_img, bb))
        if os.path.exists(sp):
            seg_img = Image.open(sp).convert('RGB')
            seg_thumbs.append(crop_to_thumb(seg_img, bb))
        else:
            seg_thumbs.append(Image.new('RGB', (THUMB, THUMB), color=(80, 80, 80)))

    strip_w = THUMB * k + PAD * (k + 1)
    strip_h = THUMB + PAD * 2
    rgb_strip = Image.new('RGB', (strip_w, strip_h), color=(20, 20, 20))
    seg_strip = Image.new('RGB', (strip_w, strip_h), color=(20, 20, 20))
    for i, (rt, st) in enumerate(zip(rgb_thumbs, seg_thumbs)):
        x = PAD + i * (THUMB + PAD)
        rgb_strip.paste(rt, (x, PAD))
        seg_strip.paste(st, (x, PAD))
        ImageDraw.Draw(rgb_strip).text(
            (x + 4, PAD + 4), f't={rec["frame_idx"] - (k - 1 - i)}',
            fill='white', font=font)

    # 3) Compose vertically (full + RGB strip + seg strip).
    canvas_w = max(full_w, strip_w)
    canvas_h = full_h + strip_h * 2 + 60
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
    canvas.paste(full, ((canvas_w - full_w) // 2, 0))
    canvas.paste(rgb_strip, ((canvas_w - strip_w) // 2, full_h + 30))
    canvas.paste(seg_strip, ((canvas_w - strip_w) // 2, full_h + 30 + strip_h + 30))
    d = ImageDraw.Draw(canvas)
    d.text(((canvas_w - strip_w) // 2 + 4, full_h + 8),
           'RGB crops (oldest -> newest)', fill='white', font=big_font)
    d.text(((canvas_w - strip_w) // 2 + 4, full_h + 30 + strip_h + 8),
           'SegFormer crops', fill='white', font=big_font)
    return canvas


def stratified_sample(records, n_inst, per_inst, seed):
    rng = random.Random(seed)
    by_label = collections.defaultdict(list)
    for r in records:
        by_label[r['label']].append(r)
    half = n_inst // 2
    out = []
    for lbl in sorted(by_label.keys()):
        recs = by_label[lbl][:]
        rng.shuffle(recs)
        # group by instance_token, take per_inst per instance
        seen = {}
        for r in recs:
            seen.setdefault(r['instance_token'], []).append(r)
            if len(seen) >= half * 2:
                break
        for inst, rs in list(seen.items())[:half]:
            out.extend(rs[:per_inst])
    rng.shuffle(out)
    return out


def main():
    args = parse_args()
    with open(args.index, 'rb') as f:
        data = pickle.load(f)

    if args.split == 'auto':
        for k in ('train', 'val', 'smoke'):
            if k in data and data[k]:
                args.split = k
                break
        else:
            print('ERROR: no non-empty split in pkl', file=sys.stderr)
            sys.exit(2)

    records = data[args.split]
    if not records:
        print(f'ERROR: split {args.split!r} is empty', file=sys.stderr)
        sys.exit(2)

    if args.stratified:
        sel = stratified_sample(records, args.num_instances, args.per_instance, args.seed)
    else:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        sel = records[:args.num_instances * args.per_instance]
    print(f'Visualising {len(sel)} records from {args.split}')

    out_pi = os.path.join(args.out_dir, 'per_instance')
    os.makedirs(out_pi, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
        big_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
        big_font = ImageFont.load_default()

    for r in sel:
        out = render_record(r, font, big_font)
        fn = f'{r["instance_token"][:8]}_f{r["frame_idx"]:02d}_l{r["label"]}.png'
        out.save(os.path.join(out_pi, fn))

    print(f'Wrote {len(sel)} viz PNGs -> {out_pi}')


if __name__ == '__main__':
    main()

"""
Precompute SegFormer (ADE20k pretrained) palette PNGs for every CAM_FRONT
keyframe referenced by the seq3 index.

Output: <seg_cache_dir>/<sample_token>.png  (palette mode, 1600x900)
        encoded labels are 0..149 (ADE20k 150 classes).

Resumable: skips files that already exist. Errors-tolerant per-image.
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq-index',
                   default='/root/IntentFormer/nuscenes/data/nuscenes_ped_intent_seq3_v2.pkl',
                   help='Pickle whose records reference sample_tokens + img_paths.')
    p.add_argument('--seg-cache-dir',
                   default='/mnt/storage/IntentFormer/seg_cache')
    p.add_argument('--model-id', default='nvidia/segformer-b0-finetuned-ade-512-512')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int, default=0,
                   help='if > 0, process only first N unique images (smoke).')
    p.add_argument('--from-smoke', action='store_true',
                   help='use the .smoke.pkl path instead of the full pkl.')
    return p.parse_args()


def collect_unique_images(pkl_path, smoke):
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    if smoke:
        records = d['smoke']
    else:
        records = d['train'] + d['val']
    seen = OrderedDict()  # sample_token -> img_path
    for r in records:
        for st, ip in zip(r['sample_tokens'], r['img_paths']):
            if st not in seen:
                seen[st] = ip
    return seen


def main():
    args = parse_args()
    pkl_path = args.seq_index
    if args.from_smoke:
        pkl_path = pkl_path.replace('.pkl', '.smoke.pkl')

    os.makedirs(args.seg_cache_dir, exist_ok=True)

    print(f'Collecting unique CAM_FRONT images from {pkl_path} ...')
    images = collect_unique_images(pkl_path, smoke=args.from_smoke)
    print(f'  {len(images)} unique sample_tokens')

    items = list(images.items())
    if args.limit > 0:
        items = items[:args.limit]
        print(f'  --limit {args.limit} -> processing first {len(items)}')

    pending = [(st, ip) for st, ip in items
               if not os.path.exists(os.path.join(args.seg_cache_dir, f'{st}.png'))]
    print(f'  {len(pending)} pending (skipping {len(items) - len(pending)} already cached)')
    if not pending:
        print('Nothing to do.')
        return

    print(f'Loading SegFormer {args.model_id} on {args.device} ...')
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    processor = SegformerImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id)
    model.eval().to(args.device)

    # ADE20k palette: standard 256-entry palette where idx 0..149 are class colors.
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 256, size=(256, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)  # background black
    palette_flat = palette.flatten().tolist()

    t0 = time.time()
    n_done = 0
    bs = args.batch_size

    for i in range(0, len(pending), bs):
        batch = pending[i:i + bs]
        try:
            pil_imgs = [Image.open(ip).convert('RGB') for _, ip in batch]
            sizes = [(im.height, im.width) for im in pil_imgs]
            inputs = processor(images=pil_imgs, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(args.device)
            with torch.no_grad():
                logits = model(pixel_values=pixel_values).logits  # (B, 150, h, w)
            for k, ((st, _), (H, W)) in enumerate(zip(batch, sizes)):
                up = torch.nn.functional.interpolate(
                    logits[k:k + 1], size=(H, W),
                    mode='bilinear', align_corners=False,
                )
                seg = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                im = Image.fromarray(seg, mode='P')
                im.putpalette(palette_flat)
                im.save(os.path.join(args.seg_cache_dir, f'{st}.png'),
                        optimize=True)
                n_done += 1
        except Exception as e:
            print(f'  ERROR on batch {i}: {e}', file=sys.stderr)

        if (i // bs) % 25 == 0 or (i + bs) >= len(pending):
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (len(pending) - n_done) / max(rate, 1e-6)
            print(f'  {n_done}/{len(pending)} done  '
                  f'({elapsed:.1f}s elapsed, {rate:.1f} img/s, ETA {eta:.0f}s)')

    print(f'Done. Wrote {n_done} PNGs to {args.seg_cache_dir}.')


if __name__ == '__main__':
    main()

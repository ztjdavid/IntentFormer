"""
TF Keras Sequence for IntentFormer-on-nuScenes (k=3).

Loads each record's k RGB crops (anchor + k-1 prior) + k seg crops + k bboxes,
returns the 4-input list expected by build_intentformer().

Bbox preprocessing matches notebook cell 12:
  squarify(box, 1, img_w) -> img.crop -> img_pad('pad_resize', 224)
Bbox feature is (cx, cy, w, h) normalized to image_w / image_h.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_W = 1600
IMAGE_H = 900


# -- Notebook helpers (cells 6, 7, 9) reproduced verbatim, with minor tweaks. ----
def bbox_sanity_check(img_size, bbox):
    img_w, img_h = img_size
    bbox = list(bbox)
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_w:
        bbox[2] = img_w - 1
    if bbox[3] >= img_h:
        bbox[3] = img_h - 1
    return bbox


def squarify(bbox, squarify_ratio, img_width):
    bbox = list(bbox)
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[2] > img_width:
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox


def img_pad(img, mode='pad_resize', size=224):
    if mode == 'warp':
        return img.resize((size, size), Image.NEAREST)
    if mode == 'same':
        return img
    img_size = img.size
    ratio = float(size) / max(img_size)
    if mode == 'pad_resize' or (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
        img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))
        img = img.resize(img_size, Image.NEAREST)
    padded = Image.new('RGB', (size, size))
    padded.paste(img, ((size - img_size[0]) // 2, (size - img_size[1]) // 2))
    return padded
# ------------------------------------------------------------------------------


def _crop_resize(pil_img, bbox, size=224):
    """squarify -> crop -> pad_resize. Returns (size, size, 3) float32 uint-scale."""
    box = list(map(float, bbox[:4]))
    box = squarify(box, 1.0, pil_img.size[0])
    box = bbox_sanity_check(pil_img.size, box)
    box = [int(round(v)) for v in box]
    box = [max(0, box[0]), max(0, box[1]),
           min(pil_img.size[0] - 1, box[2]), min(pil_img.size[1] - 1, box[3])]
    if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        # degenerate bbox -> return zeros
        return np.zeros((size, size, 3), dtype=np.float32)
    cropped = pil_img.crop(box)
    padded = img_pad(cropped, mode='pad_resize', size=size)
    return tf.keras.preprocessing.image.img_to_array(padded).astype(np.float32)


class IntentFormerSeqGen(tf.keras.utils.Sequence):
    """Yields ([RGB, mask, box, label_passthrough], y) batches."""

    def __init__(self,
                 records,
                 batch_size=8,
                 input_size=224,
                 k=3,
                 shuffle=True,
                 train=True,
                 use_seg=True,
                 normalize_rgb=True,
                 limit=None,
                 seed=42):
        if limit is not None and limit > 0:
            records = list(records[:limit])
        self.records = list(records)
        self.batch_size = batch_size
        self.input_size = input_size
        self.k = k
        self.shuffle = shuffle
        self.train = train
        self.use_seg = use_seg
        self.normalize_rgb = normalize_rgb
        self.rng = np.random.default_rng(seed)
        self._indices = np.arange(len(self.records))
        if self.shuffle:
            self.rng.shuffle(self._indices)

    def on_epoch_end(self):
        if self.shuffle:
            # Fix vs. notebook bug: use permutation, not randint (no replacement).
            self.rng.shuffle(self._indices)

    def __len__(self):
        return len(self.records) // self.batch_size

    def _load_seq(self, rec):
        """Return (rgb (k,224,224,3), seg (k,224,224,3), box (k,4), y scalar)."""
        rgb_frames = []
        seg_frames = []
        box_frames = []
        for ip, sp, b in zip(rec['img_paths'], rec['seg_paths'], rec['bboxes']):
            rgb_pil = Image.open(ip).convert('RGB')
            rgb = _crop_resize(rgb_pil, b, self.input_size)
            if self.normalize_rgb:
                rgb = rgb / 255.0
            rgb_frames.append(rgb)

            if self.use_seg and os.path.exists(sp):
                seg_pil = Image.open(sp).convert('RGB')
                seg = _crop_resize(seg_pil, b, self.input_size) / 255.0
            else:
                seg = np.zeros_like(rgb)
            seg_frames.append(seg)

            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            w = abs(b[2] - b[0])
            h = abs(b[3] - b[1])
            box_frames.append([cx / IMAGE_W, cy / IMAGE_H,
                               w / IMAGE_W, h / IMAGE_H])

        return (np.stack(rgb_frames, axis=0).astype(np.float32),
                np.stack(seg_frames, axis=0).astype(np.float32),
                np.asarray(box_frames, dtype=np.float32),
                int(rec['label']))

    def __getitem__(self, index):
        idxs = self._indices[index * self.batch_size:(index + 1) * self.batch_size]
        rgbs, segs, boxes, ys = [], [], [], []
        for i in idxs:
            rgb, seg, box, y = self._load_seq(self.records[i])
            rgbs.append(rgb)
            segs.append(seg)
            boxes.append(box)
            ys.append(y)
        X_rgb = np.stack(rgbs, axis=0)        # (B, k, 224, 224, 3)
        X_seg = np.stack(segs, axis=0)        # (B, k, 224, 224, 3)
        X_box = np.stack(boxes, axis=0)       # (B, k, 4)
        y = np.asarray(ys, dtype=np.int32)    # (B,)
        # label-passthrough input has shape (B, k, 1); model graph ignores it.
        X_label = np.tile(y.astype(np.float32)[:, None, None], (1, self.k, 1))
        return [X_rgb, X_seg, X_box, X_label], y


def load_records(pkl_path, split='train'):
    """Load records from the seq pkl. split in {'train', 'val', 'smoke'}."""
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    if split not in d:
        raise KeyError(f'split {split!r} not in pkl (keys={list(d.keys())})')
    return d[split], d.get('meta', {})

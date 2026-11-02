"""Auditable image-level splits; no claim of patient independence."""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pair(root, row, verify=True):
    root = Path(root)
    paths = [row["image"], *row["masks"]]
    if verify:
        for path in paths:
            if sha256(root / path) != row["sha256"][path]:
                raise ValueError(f"Dataset changed: {path}")
    image = np.array(Image.open(root / row["image"]).convert("RGB"))
    mask = np.zeros(image.shape[:2], dtype=bool)
    for path in row["masks"]:
        part = np.array(Image.open(root / path).convert("L")) > 0
        if part.shape != mask.shape:
            raise ValueError(f"Mask/image dimensions differ: {path}")
        mask |= part
    return image, mask


def discover(root):
    root = Path(root)
    rows, excluded, seen = [], [], {}
    for label in ("benign", "malignant"):
        if not (root / label).is_dir():
            raise ValueError(f"Missing {root / label}; point to Dataset_BUSI_with_GT")
        for image in sorted((root / label).glob("*.png")):
            if "_mask" in image.stem:
                continue
            pattern = re.compile(re.escape(image.stem) + r"_mask(?:_\d+)?\.png$")
            masks = sorted(p for p in image.parent.glob(image.stem + "_mask*.png") if pattern.fullmatch(p.name))
            if not masks:
                raise ValueError(f"Missing mask: {image}")
            paths = [image, *masks]
            row = {"id": f"{label}/{image.stem}", "label": label,
                   "image": image.relative_to(root).as_posix(),
                   "masks": [p.relative_to(root).as_posix() for p in masks],
                   "sha256": {p.relative_to(root).as_posix(): sha256(p) for p in paths}}
            pixels, union = read_pair(root, row)
            # Pixel identity catches duplicates even if PNG encoding differs.
            fingerprint = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            if fingerprint in seen:
                excluded.append({"id": row["id"], "reason": "duplicate_pixels", "retained": seen[fingerprint]})
                continue
            if not union.any():
                excluded.append({"id": row["id"], "reason": "empty_mask"})
                continue
            seen[fingerprint] = row["id"]
            row["pixel_sha256"] = fingerprint
            rows.append(row)
    return rows, excluded


def allocate(total, capacities):
    """Largest remainder proportional allocation, stable class order."""
    raw = np.array(capacities, dtype=float) * total / sum(capacities)
    counts = np.floor(raw).astype(int)
    for i in sorted(range(len(counts)), key=lambda i: (-(raw[i] - counts[i]), i))[:total - sum(counts)]:
        counts[i] += 1
    return counts.tolist()


def prepare(root, destination, size=300, seed=2026):
    if size < 20:
        raise ValueError("Use at least 20 images for three nonempty stratified splits")
    rows, excluded = discover(root)
    groups = [[r for r in rows if r["label"] == c] for c in ("benign", "malignant")]
    if min(map(len, groups)) < 3 or len(rows) < size:
        raise ValueError(f"Need {size} unique lesion images and >=3 per class; found {list(map(len, groups))}")
    rng = random.Random(seed)
    chosen = []
    counts = allocate(size, list(map(len, groups)))
    if min(counts) < 3:
        raise ValueError("Pilot too imbalanced to stratify all three splits")
    for group, count in zip(groups, counts):
        rng.shuffle(group)
        chosen.append(group[:count])
    totals = allocate(size, [70, 15, 15])
    remaining = counts[:]
    split_counts = []
    for total in totals[:2]:
        take = allocate(total, remaining)
        split_counts.append(take)
        remaining = [a - b for a, b in zip(remaining, take)]
    split_counts.append(remaining)
    if any(n < 1 for counts_ in split_counts for n in counts_):
        raise ValueError("Insufficient class support for stratified splits")
    selected = []
    for split, take in zip(("train", "val", "test"), split_counts):
        for i, n in enumerate(take):
            selected.extend(dict(row, split=split) for row in chosen[i][:n])
            chosen[i] = chosen[i][n:]
    result = {"schema": 1, "seed": seed, "size": size, "split_unit": "image",
              "normal_policy": "excluded_no_lesion_prompt", "excluded": excluded,
              "records": sorted(selected, key=lambda r: r["id"])}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace an established experimental split.
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text() != payload:
        raise FileExistsError("Manifest differs; choose a new output path")
    destination.write_text(payload)
    return result


def preprocess(image):
    # Match official MedSAM: square cubic resize, uint8 cast, min-max [0,1].
    image = resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    image = image.astype(np.float32)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def box_from_mask(mask, padding=0, rng=None):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Empty masks have no lesion box")
    h, w = mask.shape
    offsets = [padding] * 4 if rng is None else [rng.randint(0, padding) for _ in range(4)]
    return np.array([max(0, x.min() - offsets[0]), max(0, y.min() - offsets[1]),
                     min(w, x.max() + 1 + offsets[2]), min(h, y.max() + 1 + offsets[3])], dtype=np.float32)


def scale_box(box, shape):
    h, w = shape
    return np.asarray(box, dtype=np.float32) / np.array([w, h, w, h], dtype=np.float32) * 1024

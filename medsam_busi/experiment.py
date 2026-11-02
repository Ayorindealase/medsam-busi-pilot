import csv
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import sha256, read_pair, preprocess, box_from_mask, scale_box
from .model import decode, resize_logits, segmentation_loss, seed_all, load_model
from .metrics import overlap, summarize, paired_comparison

PREPROCESS_VERSION = "medsam-square-cubic-uint8-minmax-v1"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def environment():
    versions = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "scikit-image", "medsam"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"python": platform.python_version(), "packages": versions,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def cache_embeddings(root, manifest, checkpoint, output, device):
    manifest, output = Path(manifest), Path(output)
    records = json.loads(manifest.read_text())["records"]
    metadata = {"manifest_sha256": sha256(manifest), "checkpoint_sha256": sha256(checkpoint),
                "preprocess": PREPROCESS_VERSION}
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["identity"] != metadata:
        raise ValueError("Cache identity mismatch; use a new directory")
    if not meta_path.exists() and any(output.iterdir()):
        raise ValueError("Refusing untracked cache files; use an empty directory")
    write_json(meta_path, {"identity": metadata, "environment": environment()})
    model = load_model(checkpoint, device)
    for index, row in enumerate(records):
        image, _ = read_pair(root, row)  # Verify even when resuming an existing cache.
        path = cache_path(output, row)
        if not path.exists():
            tensor = torch.from_numpy(preprocess(image))[None].to(device)
            with torch.no_grad():
                embedding = model.image_encoder(tensor).cpu().numpy()
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, embedding, allow_pickle=False)
            temporary.replace(path)
        print(f"Cached {index + 1}/{len(records)}", flush=True)


def cache_path(cache, row):
    return Path(cache) / (row["pixel_sha256"] + ".npy")


def load_context(root, manifest, cache, checkpoint, device):
    identity = json.loads((Path(cache) / "metadata.json").read_text())["identity"]
    expected = {"manifest_sha256": sha256(manifest), "checkpoint_sha256": sha256(checkpoint), "preprocess": PREPROCESS_VERSION}
    if identity != expected:
        raise ValueError("Cache does not match manifest/checkpoint/preprocessing")
    records = json.loads(Path(manifest).read_text())["records"]
    ids = [r["id"] for r in records]
    hashes = [r["pixel_sha256"] for r in records]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("Manifest contains duplicate images")
    for row in records:
        read_pair(root, row)
        if not cache_path(cache, row).exists():
            raise FileNotFoundError(f"Missing cache for {row['id']}; run cache first")
    return records, identity, load_model(checkpoint, device)


def sample(root, row, cache, device, padding=0, rng=None):
    _, mask = read_pair(root, row, verify=False)
    embedding = np.load(cache_path(cache, row), allow_pickle=False)
    if embedding.shape != (1, 256, 64, 64) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid embedding: {row['id']}")
    box = scale_box(box_from_mask(mask, padding, rng), mask.shape)
    return torch.from_numpy(embedding).to(device), torch.from_numpy(box)[None].to(device), mask


def evaluate_rows(model, root, rows, cache, device, prediction_dir=None):
    model.eval()
    results = []
    with torch.no_grad():
        for row in rows:
            embedding, box, mask = sample(root, row, cache, device)
            # Official inference upsamples probabilities (not logits).
            probability = resize_logits(decode(model, embedding, box).sigmoid(), mask.shape)
            pred = probability[0, 0].cpu().numpy() > .5
            results.append({"id": row["id"], "label": row["label"], **overlap(pred, mask)})
            if prediction_dir is not None:
                path = Path(prediction_dir) / (row["id"].replace("/", "_") + ".png")
                Image.fromarray(pred.astype(np.uint8) * 255).save(path)
    return results


def train(root, manifest, cache, checkpoint, output, device="cuda", epochs=10, lr=1e-4, seed=2026, jitter=10, accumulation=4):
    if epochs < 1 or lr <= 0 or accumulation < 1 or jitter < 0:
        raise ValueError("Invalid training hyperparameters")
    seed_all(seed)
    records, identity, model = load_context(root, manifest, cache, checkpoint, device)
    training = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "val"]
    if not training or not validation:
        raise ValueError("Both train and val splits are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(epochs=epochs, lr=lr, seed=seed, jitter=jitter, accumulation=accumulation)
    write_json(output / "run.json", {"identity": identity, "config": config, "environment": environment()})
    optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=lr, weight_decay=.01)
    rng, best, history = random.Random(seed), -1., []
    for epoch in range(epochs):
        model.eval()
        model.mask_decoder.train()
        rng.shuffle(training)
        losses = []
        # Explicit microbatch groups correctly normalize the short final group.
        for start in range(0, len(training), accumulation):
            group = training[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for row in group:
                embedding, box, mask = sample(root, row, cache, device, jitter, rng)
                target = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
                loss = segmentation_loss(decode(model, embedding, box), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            optimizer.step()
        val = summarize(evaluate_rows(model, root, validation, cache, device))["all"]["dice"]
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_dice": val}
        history.append(record)
        write_json(output / "history.json", history)
        if val > best:
            best = val
            torch.save({"decoder": model.mask_decoder.state_dict(), "identity": identity,
                        "config": config, "epoch": epoch + 1, "val_dice": val}, output / "best_decoder.pt")
        print(json.dumps(record), flush=True)
    return history


def evaluate(root, manifest, cache, checkpoint, output, device="cuda", decoder=None, split="test"):
    records, identity, model = load_context(root, manifest, cache, checkpoint, device)
    if decoder:
        state = torch.load(decoder, map_location=device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Decoder checkpoint provenance differs")
        model.mask_decoder.load_state_dict(state["decoder"], strict=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    rows = evaluate_rows(model, root, [r for r in records if r["split"] == split], cache, device, output / "predictions")
    report = {"identity": identity, "split": split, "prompt": "tight_GT_union_box_xyxy_exclusive_v1",
              "threshold": .5, "decoder_sha256": sha256(decoder) if decoder else None,
              "summary": summarize(rows), "per_image": rows, "environment": environment()}
    write_json(output / "metrics.json", report)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "label", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    return report


def compare(baseline_path, tuned_path, output):
    a, b = [json.loads(Path(p).read_text()) for p in (baseline_path, tuned_path)]
    for key in ("identity", "split", "prompt", "threshold"):
        if a[key] != b[key]:
            raise ValueError(f"Evaluation protocols differ: {key}")
    if a["decoder_sha256"] is not None or b["decoder_sha256"] is None:
        raise ValueError("Expected baseline first and fine-tuned report second")
    result = {"baseline": a["summary"], "fine_tuned": b["summary"],
              "paired_change": paired_comparison(a["per_image"], b["per_image"]),
              "limitation": "Image bootstrap; patient dependence is unknown. GT boxes are oracle prompts."}
    write_json(output, result)
    return result

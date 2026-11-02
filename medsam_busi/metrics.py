import numpy as np


def overlap(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    intersection = int((prediction & target).sum())
    total = int(prediction.sum() + target.sum())
    union = int((prediction | target).sum())
    return {"dice": 2 * intersection / total if total else 1.0,
            "iou": intersection / union if union else 1.0}


def summarize(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    return {label: {"n": len(subset), **{metric: float(np.mean([r[metric] for r in subset]))
            for metric in ("dice", "iou")}}
            for label in ("all", "benign", "malignant")
            if (subset := [r for r in rows if label == "all" or r["label"] == label])}


def paired_comparison(baseline, tuned, seed=2026, repeats=2000):
    a = {r["id"]: r for r in baseline}
    b = {r["id"]: r for r in tuned}
    if not a or a.keys() != b.keys() or len(a) != len(baseline) or len(b) != len(tuned):
        raise ValueError("Comparison requires identical, unique image IDs")
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ("dice", "iou"):
        delta = np.array([b[k][metric] - a[k][metric] for k in sorted(a)])
        boot = rng.choice(delta, size=(repeats, len(delta)), replace=True).mean(1)
        result[metric] = {"mean_delta": float(delta.mean()), "image_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist()}
    return result

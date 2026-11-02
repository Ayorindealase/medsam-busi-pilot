import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from medsam_busi.data import prepare, discover, read_pair, preprocess, box_from_mask, scale_box, sha256
from medsam_busi.metrics import overlap, summarize, paired_comparison
from medsam_busi.model import freeze_encoders, decode, segmentation_loss
from medsam_busi import experiment


@pytest.fixture
def busi(tmp_path):
    root = tmp_path / "busi"
    rng = np.random.default_rng(12)
    for label in ("benign", "malignant"):
        (root / label).mkdir(parents=True)
        for i in range(20):
            image = rng.integers(0, 256, (12, 18), dtype=np.uint8)
            mask = np.zeros_like(image)
            mask[2:7, 4:10] = 255
            Image.fromarray(image).save(root / label / f"{label} ({i}).png")
            Image.fromarray(mask).save(root / label / f"{label} ({i})_mask.png")
    return root


def test_split_reproducible_disjoint_and_stratified(busi, tmp_path):
    a = prepare(busi, tmp_path / "a.json", 20)
    b = prepare(busi, tmp_path / "b.json", 20)
    assert a == b
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert len({r["id"] for r in a["records"]}) == 20
    assert [sum(r["split"] == s for r in a["records"]) for s in ("train", "val", "test")] == [14, 3, 3]
    for split in ("train", "val", "test"):
        assert {r["label"] for r in a["records"] if r["split"] == split} == {"benign", "malignant"}
    with pytest.raises(FileExistsError):
        prepare(busi, tmp_path / "a.json", 20, seed=7)
    with pytest.raises(ValueError):
        prepare(busi, tmp_path / "c.json", 300)


def test_300_counts(monkeypatch, tmp_path):
    rows = [{"id": str(i), "label": "benign" if i < 400 else "malignant"} for i in range(600)]
    monkeypatch.setattr("medsam_busi.data.discover", lambda _: (rows, []))
    data = prepare("unused", tmp_path / "pilot.json")
    assert [sum(r["split"] == s for r in data["records"]) for s in ("train", "val", "test")] == [210, 45, 45]


def test_multimask_duplicates_and_tampering(busi):
    image = busi / "benign/benign (0).png"
    Image.open(image).save(busi / "benign/duplicate.png")
    Image.open(busi / "benign/benign (0)_mask.png").save(busi / "benign/duplicate_mask.png")
    extra = np.zeros((12, 18), np.uint8)
    extra[10, 16] = 255
    Image.fromarray(extra).save(busi / "benign/benign (0)_mask_1.png")
    rows, excluded = discover(busi)
    assert len(rows) == 40 and len(excluded) == 1
    row = next(r for r in rows if r["id"] == "benign/benign (0)")
    _, union = read_pair(busi, row)
    assert union[10, 16] and union.sum() == 31
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset changed"):
        read_pair(busi, row)


def test_missing_mask_fails(busi):
    (busi / "benign/benign (0)_mask.png").unlink()
    with pytest.raises(ValueError, match="Missing mask"):
        discover(busi)


def test_preprocessing_boxes_and_empty():
    mask = np.zeros((10, 20), bool)
    mask[2:5, 4:8] = True
    np.testing.assert_equal(box_from_mask(mask), [4, 2, 8, 5])
    np.testing.assert_allclose(scale_box(box_from_mask(mask), mask.shape), [204.8, 204.8, 409.6, 512])
    np.testing.assert_equal(box_from_mask(mask, 100), [0, 0, 20, 10])
    assert np.array_equal(box_from_mask(mask, 10, random.Random(1)), box_from_mask(mask, 10, random.Random(1)))
    with pytest.raises(ValueError):
        box_from_mask(np.zeros((10, 20), bool))
    output = preprocess(np.ones((10, 20, 3), np.uint8) * 128)
    assert output.shape == (3, 1024, 1024) and np.isfinite(output).all() and not output.any()


def test_metrics_and_pairing():
    a = np.array([[1, 0], [0, 0]], bool)
    assert overlap(a, a) == {"dice": 1, "iou": 1}
    assert overlap(~a, a) == {"dice": 0, "iou": 0}
    assert overlap(a * 0, a * 0)["dice"] == 1
    assert overlap(a * 0, a)["dice"] == 0
    rows = [{"id": "a", "label": "benign", "dice": .5, "iou": .3}]
    assert summarize(rows)["all"]["n"] == 1
    assert paired_comparison(rows, rows)["dice"]["mean_delta"] == 0
    with pytest.raises(ValueError):
        paired_comparison(rows, [])


class Prompt(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
    def forward(self, points, boxes, masks):
        return boxes.mean() * self.weight, self.weight
    def get_dense_pe(self):
        return self.weight


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 1, 1)
    def forward(self, image_embeddings, **kwargs):
        return self.conv(image_embeddings), None


class TinySAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Conv2d(3, 256, 1)
        self.prompt_encoder = Prompt()
        self.mask_decoder = Decoder()


def test_frozen_gradient_contract():
    model = freeze_encoders(TinySAM())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=.01)
    logits = decode(model, torch.ones(1, 256, 4, 4), torch.ones(1, 4))
    segmentation_loss(logits, torch.ones(1, 1, 8, 8)).backward()
    optimizer.step()
    assert all(p.grad is None for p in model.image_encoder.parameters())
    assert all(p.grad is None for p in model.prompt_encoder.parameters())
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith("mask_decoder"))
    with pytest.raises(ValueError):
        decode(model, torch.ones(2, 256, 4, 4), torch.ones(2, 4))


def test_end_to_end_cached_synthetic(busi, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    manifest = tmp_path / "manifest.json"
    data = prepare(busi, manifest, 20)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_bytes(b"synthetic fixture only")
    cache = tmp_path / "cache"
    cache.mkdir()
    identity = {"manifest_sha256": sha256(manifest), "checkpoint_sha256": sha256(checkpoint), "preprocess": experiment.PREPROCESS_VERSION}
    experiment.write_json(cache / "metadata.json", {"identity": identity})
    for row in data["records"]:
        np.save(experiment.cache_path(cache, row), np.ones((1, 256, 64, 64), np.float32))
    def load(*args):
        torch.manual_seed(3)
        return freeze_encoders(TinySAM())
    monkeypatch.setattr(experiment, "load_model", load)
    common = dict(root=busi, manifest=manifest, cache=cache, checkpoint=checkpoint, device="cpu")
    baseline = experiment.evaluate(**common, output=tmp_path / "baseline")
    history = experiment.train(**common, output=tmp_path / "train", epochs=2)
    assert len(history) == 2
    tuned = experiment.evaluate(**common, output=tmp_path / "tuned", decoder=tmp_path / "train/best_decoder.pt")
    assert baseline["per_image"][0]["id"] == tuned["per_image"][0]["id"]
    experiment.compare(tmp_path / "baseline/metrics.json", tmp_path / "tuned/metrics.json", tmp_path / "comparison.json")
    identity["checkpoint_sha256"] = "wrong"
    experiment.write_json(cache / "metadata.json", {"identity": identity})
    with pytest.raises(ValueError, match="Cache does not match"):
        experiment.evaluate(**common, output=tmp_path / "bad")


@pytest.mark.integration
def test_official_decoder_step():
    pytest.importorskip("segment_anything")
    from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
    torch.set_num_threads(1)
    model = TinySAM()
    model.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16)
    model.mask_decoder = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8), transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
    freeze_encoders(model)
    before = model.mask_decoder.output_hypernetworks_mlps[0].layers[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.mask_decoder.parameters(), lr=1e-4)
    logits = decode(model, torch.randn(1, 256, 64, 64), torch.tensor([[20., 30., 400., 500.]]))
    assert logits.shape == (1, 1, 256, 256)
    loss = segmentation_loss(logits, torch.ones(1, 1, 35, 47))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.mask_decoder.output_hypernetworks_mlps[0].layers[0].weight)
    assert all(p.grad is None for p in model.prompt_encoder.parameters())

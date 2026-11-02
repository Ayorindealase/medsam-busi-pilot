# Validation record — 2026-11-02

## Executed locally

- **10 pytest tests passed** on CPU, including notebook schema validation and Python syntax checks for every notebook code cell.
- Deterministic manifests; stratification; disjoint IDs; exact 210/45/45 allocation for a synthetic 300-record fixture; protection against accidental split overwrites.
- Multi-mask union, pixel-duplicate exclusion, missing-mask errors, and file-tampering detection.
- Constant-image normalization, box coordinates, clipping, scaling, seeded jitter and empty-mask handling.
- Dice/IoU edge cases and strict paired ID matching.
- Frozen encoder gradient checks and decoder parameter updates.
- Synthetic BUSI-shaped data through cached baseline evaluation, two training epochs, validation selection, decoder save/load, tuned evaluation and paired comparison; cache-provenance mismatch rejection.
- Official pinned MedSAM prompt encoder and mask decoder forward/backward/optimizer step on synthetic embeddings (included in the 10 passing tests; not skipped).
- Separate full ViT-B structural smoke: random checkpoint round trip through the official loader, 1024×1024 image encoder forward, prompt encoding and mask decoding. Passed, with finite output of shape `(1, 1, 256, 256)`.

Commands, from the source root after installation:

```bash
python -m pytest -q
python tests/smoke_full_model.py
```

The local test environment used Python 3.11, PyTorch 2.0.1, NumPy 1.26.4, scikit-image 0.22.0 and MedSAM revision `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488`. Isolated temporary dependencies were used to resolve an existing local NumPy binary compatibility problem. Colab requirements pin newer Pillow/scikit-image versions and retain the runtime's matching CUDA torch/torchvision pair; that exact hosted environment has not been executed here. A torchvision optional image-extension warning appeared locally; this project reads images using Pillow, and all checks passed.

## Not yet verified

No BUSI images or official pretrained MedSAM weights were available in this workspace. No real 300-image manifest, pretrained inference scores, fine-tuned BUSI scores, training-time estimate, GPU memory measurement or executed Colab run is claimed. Random-weight smoke results are structural checks only. The first GitHub Actions run failed during upstream MedSAM installation. The dependency now pins revision `b9db486e77bf9a2c9c11cc061c63f54110d520b6`, before the custom installer that attempted a conflicting nested MONAI/NumPy installation, and pins MONAI 1.4.0. Upstream comparison confirmed only `setup.py` differs between these MedSAM revisions; architecture files are unchanged. See the repository Actions page for the corrected run result.

The first real run must verify source archive contents, inspect the manifest and exclusions, preview training image/mask alignment, complete embedding caching on the selected GPU, train/select using validation only, and run the fixed test comparison once. Review exported provenance and results before team reporting.

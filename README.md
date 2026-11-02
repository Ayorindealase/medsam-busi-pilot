# MedSAM on BUSI: 300-image Colab pilot

A small-scale reproduction of the **box-prompted MedSAM workflow**, comparing pretrained MedSAM with decoder-only adaptation on BUSI. This is not a reproduction of the paper's large-scale training or reported performance. No BUSI performance results are claimed yet.

## What this project does

Read the [project overview](docs/PROJECT_OVERVIEW.md) for a first-person explanation of what I built, the experiment design, completed checks, and the work still needed before reporting BUSI results.

## Test on your own computer

Follow the [team testing guide](docs/TEAM_TESTING_GUIDE.md) for Windows, macOS and Linux setup, software-only checks, the full BUSI workflow, troubleshooting and reporting results.

## Start in Colab

Open the [pilot notebook in Colab](https://colab.research.google.com/github/Ayorindealase/medsam-busi-pilot/blob/main/notebooks/BUSI_MedSAM_Pilot.ipynb). Select a GPU runtime (a T4 is the intended starting point). Run the notebook from the top; its setup cell clones this repository into `/content/medsam-busi-pilot` when needed. A source ZIP upload fallback is also available. The notebook includes this setup option, Drive mounting, file checks, tests, preparation, caching, training, paired evaluation, previews, and result export.

Obtain the original BUSI dataset through the [dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/) and its linked author source, observing the source's usage terms. Extract it yourself to a folder named `Dataset_BUSI_with_GT`; do not use BUSI_WHU or a pre-split derivative. Obtain `medsam_vit_b.pth` from the checkpoint link in the [official MedSAM README](https://github.com/bowang-lab/MedSAM#get-started) (navigate to Get Started). Place both in Google Drive and update the notebook paths. No credentials, data, or weights are included in this repository. Only load checkpoints from trusted sources.

Expected dataset layout:

```text
Dataset_BUSI_with_GT/
  benign/benign (1).png
  benign/benign (1)_mask.png
  benign/benign (1)_mask_1.png  # where present
  malignant/malignant (1).png
  malignant/malignant (1)_mask.png
  normal/...
```

## Experiment protocol

- Sample 300 unique lesion-bearing images, proportionally stratified by benign/malignant class, seed 2026. Split 210 train / 45 validation / 45 test. Persist selected paths, class, split, file hashes, and exclusions in a JSON manifest. Counts depend on actual archive contents, not assumed published class counts.
- Union all masks for each image, producing one lesion-union segmentation and enclosing box. Exclude normal/empty masks and remove exact pixel duplicates before sampling. Missing masks and size mismatches fail explicitly. Near-duplicates are not detected; review the dataset and errata before interpreting results.
- This is an **image-level split**. BUSI filenames do not provide reliable patient identities; patient independence is not established. The pretrained model's possible prior exposure to BUSI is also unverified. Do not label results as external validation or guaranteed zero-shot performance.
- Match MedSAM's square 1024×1024 cubic resize, uint8 conversion, and min-max normalization. Call its encoder directly, avoiding SAM's separate normalization/padding path.
- Start both arms from the same MedSAM ViT-B checkpoint. Freeze image and prompt encoders. Cache float32 image embeddings once; approximately 1.2 GiB for 300 images, plus source data and weights.
- Train only the mask decoder for 10 epochs using AdamW (learning rate 1e-4, weight decay .01), BCE plus squared-denominator soft Dice, one image per decoder call and accumulation over four images. Jitter train boxes outward by 0–10 original-image pixels per side. No image augmentation is used because embeddings are cached.
- Choose the best epoch by macro validation Dice using tight boxes. Evaluate baseline and that selected decoder on the same test manifest, with tight annotation-derived boxes and threshold 0.5. These are **oracle box prompts**, so this is prompted segmentation, not automatic lesion detection. Test masks supply the simulated prompts and scoring targets, never optimization or checkpoint selection.
- Save original-resolution predictions, per-image Dice/IoU, overall and per-class macro means, and paired deltas with image-bootstrap intervals. Image bootstrap does not account for unknown patient clustering.

## Local commands

Use Python 3.10–3.12 and a matching PyTorch/torchvision installation; the notebook retains Colab's CUDA pair. Other dependencies and the MedSAM source revision are pinned in `requirements-colab.txt`.

```bash
python -m pip install -r requirements-colab.txt
python -m pytest -q
python -m medsam_busi.cli prepare --root data/Dataset_BUSI_with_GT --output artifacts/manifest.json
python -m medsam_busi.cli cache --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --checkpoint checkpoints/medsam_vit_b.pth --output artifacts/cache
python -m medsam_busi.cli train --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --cache artifacts/cache --checkpoint checkpoints/medsam_vit_b.pth --output runs/train
python -m medsam_busi.cli evaluate --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --cache artifacts/cache --checkpoint checkpoints/medsam_vit_b.pth --output runs/baseline
python -m medsam_busi.cli evaluate --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --cache artifacts/cache --checkpoint checkpoints/medsam_vit_b.pth --decoder runs/train/best_decoder.pt --output runs/tuned
python -m medsam_busi.cli compare --baseline-path runs/baseline/metrics.json --tuned-path runs/tuned/metrics.json --output runs/comparison.json
```

Use `--device cpu` for functional debugging; a GPU is strongly preferred for the real encoder pass. The embedding cache resumes after interruption. Training starts fresh and deliberately refuses existing output directories; choose a new run name. Change hyperparameters using validation only, then perform one final test comparison. Record all trials. Exact bitwise CUDA reproducibility across runtimes is not promised: seeds and deterministic settings are applied, and versions/device/checkpoint/manifest hashes are logged.

## Validation and sharing

See [validation](docs/VALIDATION.md) for executed checks and remaining GPU/data validation, [publication](docs/PUBLICATION.md) for the selected GitHub destination, and [protocol notes](docs/PROTOCOL.md) for deviations and limitations. Data, caches, weights, predictions, and generated runs are ignored by Git. Share source and approved aggregate results; do not commit raw datasets or weights.

## Attribution

- Ma et al., *Segment anything in medical images*, Nature Communications 15, 654 (2024), [paper](https://doi.org/10.1038/s41467-024-44824-z), [code](https://github.com/bowang-lab/MedSAM).
- Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief 28, 104863 (2020), [paper](https://doi.org/10.1016/j.dib.2019.104863).
- [BUSI errata letter](https://pubmed.ncbi.nlm.nih.gov/37383756/).

The upstream architecture is installed as a dependency, not vendored. Upstream software, pretrained weights, and dataset terms remain separate. No open-source license is assigned to this pilot yet; public visibility alone does not grant a reuse license. Research use only; not a clinical diagnostic tool.

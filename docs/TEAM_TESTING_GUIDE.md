# Team guide: test the MedSAM BUSI pilot on your computer

Repository: https://github.com/Ayorindealase/medsam-busi-pilot

Start with the software checks in steps 1–4. They use synthetic data, so you do **not** need BUSI images, model weights, a GPU, or a Google account. Once those checks pass, steps 5–8 explain how to run the real 300-image pilot.

The goal is to check that the pipeline works and then compare pretrained MedSAM with decoder-only fine-tuning. Passing software tests does not establish segmentation accuracy. Real BUSI results and a full Colab GPU run have not yet been validated.

## 1. Install the prerequisites

Use **Python 3.11**, Git, and an internet connection. Open Terminal on macOS/Linux or PowerShell on Windows. Run commands from the same terminal session, one line at a time.

CPU testing is supported. For the full experiment, use an NVIDIA GPU with a compatible PyTorch installation or the [Colab notebook](https://colab.research.google.com/github/Ayorindealase/medsam-busi-pilot/blob/main/notebooks/BUSI_MedSAM_Pilot.ipynb). On a Mac, use CPU for these checks; Apple MPS acceleration has not been validated. Windows and macOS instructions are provided, but those exact fresh-install paths have not been independently tested on every platform.

Allow several GB for software, weights and data. The 300-image embedding cache alone is about 1.2 GiB. Actual RAM/VRAM requirements and runtime have not been benchmarked; the full model smoke test needs more memory than the regular tests.

## 2. Download the project and create an isolated environment

On every platform:

```bash
git clone https://github.com/Ayorindealase/medsam-busi-pilot.git
cd medsam-busi-pilot
```

If you already have a clone with no local edits, use `git pull --ff-only` from its folder instead.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If your organization's policy blocks activation, do not change that policy. Run the Python commands below using `.\.venv\Scripts\python.exe` in place of `python`; activation is optional.

**macOS/Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

After activation, on every platform:

```bash
python --version
python -m pip install --upgrade pip
```

The version should begin with `Python 3.11`. Use the environment's Python for every subsequent command.

## 3. Install PyTorch and project dependencies

Choose **one** PyTorch command below. The version pair matches the repository's Linux CI configuration. These install choices follow the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v251).

**Windows/Linux, CPU testing:**

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

**macOS, CPU testing:**

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1
```

If no wheel is available for your Mac architecture, use a supported Python/platform combination or Colab; do not attempt to build PyTorch from source just to test this pilot.

**Windows/Linux, compatible NVIDIA GPU:**

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

The GPU choice requires a driver compatible with the CUDA 12.1 build. If your hardware needs a newer build, use the official PyTorch selector for a matching torch/torchvision pair and record those versions as a different test environment.

Then, on every platform:

```bash
python -m pip install -r requirements-colab.txt
python -m pip check
python -c "import torch, torchvision, segment_anything; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Despite its filename, `requirements-colab.txt` is also used for local setup. It installs the project and its pinned dependencies while retaining a compatible installed torch/torchvision pair. `CUDA available: False` is expected on CPU-only computers and Macs. An NVIDIA GPU run requires `True`.

The MedSAM source is pinned to a revision before an incompatible custom installer; the architecture files are unchanged between that revision and the original pilot pin. MONAI is pinned because it is an upstream dependency, although this pilot implements its own loss.

## 4. Run the software tests — no dataset needed

From the repository root:

```bash
python -m pytest -q
```

Expected result for this version: **10 passed**. A skipped official-decoder integration test means the MedSAM dependency is missing; it is not a complete pass. Send us the output if anything fails.

For an additional full-model check:

```bash
python tests/smoke_full_model.py
```

Expected output includes:

```text
'random_vit_b_checkpoint_roundtrip_and_forward': 'passed'
```

This check loads a temporary checkpoint, processes a 1024×1024 synthetic image through ViT-B, and decodes a mask. It runs on CPU, uses random weights, and deletes its temporary checkpoint afterward. It does not download trained weights and does not measure medical accuracy.

**Stop here if your assignment is to verify installation and software correctness.** Report the details in step 9. Continue only if you are running the real dataset experiment.

## 5. Add the BUSI dataset and pretrained MedSAM checkpoint

Obtain the original BUSI dataset through the [BUSI article and its linked author source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/), observing the source's terms. Download `medsam_vit_b.pth` through **Get Started** in the [official MedSAM repository](https://github.com/bowang-lab/MedSAM#get-started). Do not substitute the original SAM checkpoint, LiteMedSAM, MedSAM2, BUSI_WHU, or a pre-split BUSI derivative.

Using your file manager, create these folders inside your local clone and put the extracted files in this layout:

```text
medsam-busi-pilot/
  data/
    Dataset_BUSI_with_GT/
      benign/
        benign (1).png
        benign (1)_mask.png
        ...
      malignant/
        malignant (1).png
        malignant (1)_mask.png
        ...
      normal/
        ...
  checkpoints/
    medsam_vit_b.pth
```

Keep additional masks such as `benign (1)_mask_1.png`; the pipeline combines them. Check for an accidentally nested `Dataset_BUSI_with_GT/Dataset_BUSI_with_GT` directory. Only load weights from trusted sources.

Team members comparing results must use the same original archive, checkpoint and source commit. The pipeline records hashes and rejects changed files. Raw data and weights are ignored by Git and should remain local.

## 6. Prepare the fixed 300-image pilot

```bash
python -m medsam_busi.cli prepare --root data/Dataset_BUSI_with_GT --output artifacts/manifest.json --size 300 --seed 2026
```

Expected output includes `"size": 300` and `"seed": 2026`. Review `artifacts/manifest.json`; it records selected images, labels, train/validation/test assignment, hashes and exclusions. The split contains **210 training, 45 validation and 45 test images**. Selection is proportional by benign/malignant class after exclusions.

Normal images are excluded because this experiment requires a lesion prompt. Exact pixel duplicates and empty masks are excluded; missing masks and shape mismatches cause errors. If fewer than 300 eligible unique images remain, investigate the archive rather than changing the selection silently.

Before training, inspect several **training** images and all their masks together to confirm alignment and annotation quality. The Colab notebook contains a training-only preview cell if you prefer a visual workflow. Keep the manifest unchanged across baseline and fine-tuning; use a new manifest path for a genuinely different experiment.

## 7. Cache embeddings and train

The following commands use an NVIDIA GPU. If you are deliberately debugging on CPU, replace **every** `--device cuda` with `--device cpu`; processing the whole dataset on CPU may be slow.

First cache the frozen image-encoder outputs:

```bash
python -m medsam_busi.cli cache --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --checkpoint checkpoints/medsam_vit_b.pth --output artifacts/cache --device cuda
```

It should finish with `Cached 300/300`. You can repeat the same command after interruption to reuse completed entries. Do not reuse this cache with another manifest or starting checkpoint.

Train the decoder:

```bash
python -m medsam_busi.cli train --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --cache artifacts/cache --checkpoint checkpoints/medsam_vit_b.pth --output runs/train --device cuda --epochs 10 --lr 0.0001 --seed 2026 --accumulation 4
```

Expected files include `runs/train/run.json`, `history.json`, and `best_decoder.pt`. The best checkpoint is chosen by validation Dice. Check that losses and validation values are finite. An existing output directory causes an error intentionally: choose a new name such as `runs/train_trial02` for another trial. Training does not resume from an interrupted epoch; only the embedding cache is resumable.

If you want a one-epoch installation trial, use a separate output such as `runs/smoke_train` and `--epochs 1`. Inspect validation results only, then start the planned 10-epoch run from the original pretrained weights. Do not report the installation trial as the final pilot.

## 8. Evaluate and compare

Run this section after fixing the training settings. Do not use test scores to choose hyperparameters or epochs.

Pretrained baseline:

```bash
python -m medsam_busi.cli evaluate --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --cache artifacts/cache --checkpoint checkpoints/medsam_vit_b.pth --output runs/baseline --device cuda
```

Selected fine-tuned decoder:

```bash
python -m medsam_busi.cli evaluate --root data/Dataset_BUSI_with_GT --manifest artifacts/manifest.json --cache artifacts/cache --checkpoint checkpoints/medsam_vit_b.pth --decoder runs/train/best_decoder.pt --output runs/tuned --device cuda
```

Compare the two:

```bash
python -m medsam_busi.cli compare --baseline-path runs/baseline/metrics.json --tuned-path runs/tuned/metrics.json --output runs/comparison.json
```

Each evaluation folder contains `metrics.json`, `per_image.csv`, and original-resolution mask PNGs in `predictions/`. `runs/comparison.json` summarizes baseline and fine-tuned Dice/IoU and paired changes. Larger Dice/IoU is better; a positive paired delta means the fine-tuned arm scored higher. An improvement is not guaranteed.

Boxes are derived from ground-truth masks in both arms. This measures **box-prompted segmentation**, not automatic detection. Splits are image-level; patient independence and pretraining overlap are unverified. The bootstrap intervals treat images as independent and do not resolve these limitations.

## 9. Send your test report to the team

Record the code and environment before reporting:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > environment.txt
```

Include:

- Your operating system, processor, RAM, GPU model if any, and the commit hash.
- Whether you ran software tests, the full-model smoke check, or the real BUSI pilot.
- The test summary and any complete error output.
- For real runs: manifest, `train/run.json`, `train/history.json`, both evaluation `metrics.json` and `per_image.csv` files, and `comparison.json`.
- Any changes to commands, settings, package versions or source data.

Send `environment.txt` with the report; keep it out of the repository and check it for local paths before sharing. Do not include the dataset, model weights, cached embeddings or patient images in a public GitHub issue. Coordinate any prediction-image sharing with the team.

A useful report starts: “I tested commit ___ on ___ using CPU/GPU ___. Software tests: ___. Full-model check: ___. Real BUSI run: not run/completed/failed at ___. Attached are ___.”

## Common problems

**`python` or `git` is not recognized:** install the prerequisite, reopen your terminal, and confirm Python 3.11 is selected.

**`No module named medsam_busi` or `segment_anything`:** activate the project environment, change to the repository root, and repeat the dependency installation using `python -m pip`.

**NumPy binary incompatibility or an upstream installer failure:** pull the latest source and start with a fresh virtual environment. The initial MedSAM dependency pin had an incompatible installer; the current pin fixes that path. Avoid mixing this environment with an existing research environment.

**CUDA is unavailable:** CPU-only installs and Macs should use `--device cpu`. For NVIDIA, check the driver and chosen PyTorch wheel. Do not run the GPU commands until the availability check returns `True`.

**Out of memory:** close other GPU workloads; use CPU for functional checks or move the full run to Colab. The decoder already processes one image at a time. Changing gradient accumulation does not reduce the memory needed for one encoder image.

**Missing files or mismatched masks:** verify the folder layout, original filenames, image dimensions and checkpoint path. Preserve every additional mask.

**Cache identity mismatch or dataset changed:** restore the exact original inputs, or use a new manifest and cache directory for a new experiment. Do not edit hashes to bypass the checks.

**Output directory already exists:** choose a fresh run/evaluation directory and update all downstream paths consistently. Preserve old outputs if they are part of a reported experiment.

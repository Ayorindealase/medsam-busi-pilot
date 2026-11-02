# What I built: a small-scale MedSAM pilot on BUSI

**Project update — November 2, 2026**

I prepared a reproducible research pipeline to test MedSAM on a planned 300-image subset of the BUSI breast ultrasound dataset. The project includes data preparation, pretrained-model inference, decoder-only fine-tuning, evaluation, automated tests, and a Google Colab notebook. I published the code so the team can inspect it, test it on their own computers, and run the same experiment.

The software checks have passed. The real BUSI experiment is the next stage: I have not yet established baseline accuracy, fine-tuned accuracy, or an improvement from training.

Repository: [Ayorindealase/medsam-busi-pilot](https://github.com/Ayorindealase/medsam-busi-pilot)

## Why I started this project

I wanted to turn the MedSAM paper into a manageable experiment that our team could reproduce with a smaller dataset and accessible computing resources. The question I designed the pilot to investigate is:

> Does fine-tuning only the mask decoder improve MedSAM's box-prompted segmentation on a small BUSI subset compared with the original pretrained MedSAM checkpoint?

The pilot uses the authors' existing MedSAM architecture and pretrained weights. My contribution is the BUSI experiment pipeline, its reproducibility controls, the comparison workflow, and the materials that make it easier for our team to run. I did not train the original MedSAM foundation model or recreate the paper's large-scale training study.

## What MedSAM does in this experiment

For each image, the model receives an ultrasound image and a rectangular box around the region of interest. It predicts a segmentation mask: a map indicating which pixels belong to the target region.

The model has three relevant parts. The **image encoder** converts the image into numerical features. The **prompt encoder** represents the box supplied to the model. The **mask decoder** combines that information to predict the segmentation.

I configured the pilot to keep both encoders fixed and update only the mask decoder. This reduces the amount of training computation. It also lets us calculate the image features once and reuse them throughout training instead of repeatedly processing every image through the large image encoder.

## How I designed the data preparation

I chose BUSI for the pilot and implemented a preparation step that will select **300 unique lesion-bearing images** from the supplied original archive. The selection is proportional to the available benign and malignant classes after exclusions, using a fixed random seed of 2026.

The planned split is:

- **210 training images:** used to update the decoder.
- **45 validation images:** used to select the best training checkpoint.
- **45 test images:** reserved for the final baseline-versus-fine-tuned comparison.

These counts describe the implemented protocol. The actual BUSI selection and manifest will be generated when the dataset is supplied.

For images with multiple annotation masks, the pipeline merges those masks into one target and places one enclosing box around it. Normal images are excluded because this particular experiment requires a lesion box. Empty masks are excluded, exact duplicate images are removed, and missing masks or mismatched image dimensions produce an error.

The preparation step saves a **manifest**, which is a record of the selected files, their class labels, their split assignments, file checksums and exclusions. Checksums help detect whether an input file has changed. This allows teammates to verify that they are comparing results from the same inputs rather than accidentally using different versions of the dataset.

## What I implemented for inference and training

I implemented preprocessing consistent with the official MedSAM inference workflow: images are resized to 1024 × 1024 and normalized before being passed to the image encoder. Bounding-box coordinates are scaled to match that image size.

The **baseline** uses the pretrained MedSAM checkpoint without BUSI fine-tuning. The **adapted model** starts from that same checkpoint and trains only the mask decoder.

The default training configuration runs for 10 epochs with AdamW and a learning rate of 0.0001. Its loss combines binary cross-entropy and a soft Dice term, which together guide the model toward the annotated foreground mask. Each decoder call processes one image, and gradients are accumulated across four images before an update. Training boxes receive small outward random adjustments to introduce variation in the prompts.

The image-feature cache can resume after interruption. Training itself starts fresh; checkpoint-based training resume is not implemented. The best decoder is saved according to validation Dice, so the test set does not determine which epoch we select.

## How the comparison will work

I implemented evaluation so that both models use the same test images, the same box policy, and the same prediction threshold. Predictions are evaluated at the original image dimensions.

The main metrics are **Dice** and **intersection over union (IoU)**. Both measure overlap between the predicted mask and the reference annotation; higher values indicate better overlap. The output includes per-image scores, overall averages, and separate benign/malignant averages.

The comparison also reports the score change for each matched image and a bootstrap interval for the average change. A positive change would favor the fine-tuned model, but the pipeline does not assume or guarantee an improvement.

An important detail is that the evaluation boxes are generated from the reference masks. They simulate an accurately supplied prompt. This experiment therefore measures segmentation given a box; it does not test whether the system can independently find a lesion in an unmarked image.

## What I built for the team

I published the project in a public GitHub repository with a Colab notebook and command-line tools. The notebook covers setup, Google Drive paths, data preparation, annotation previews, feature caching, training, evaluation and report export.

I also wrote a [team testing guide](TEAM_TESTING_GUIDE.md) with Windows, macOS and Linux setup instructions. Team members can first run software tests without downloading BUSI or pretrained weights, then proceed to the real experiment if they have the required files and computing resources.

The repository keeps raw data, weights, cached features and generated experiment outputs out of version control. It records training settings and software/device information with the run outputs so results can be traced back to their configuration.

## What has been tested so far

The **10 automated tests passed locally**. They cover reproducible splits, duplicate handling, mask merging, preprocessing, box coordinates, metric calculations, encoder freezing, decoder updates, checkpoint handling and notebook structure. They also exercise a complete synthetic training-and-evaluation workflow and an update through the official MedSAM decoder.

A separate structural test passed through the full ViT-B model using random weights. It checked checkpoint loading, image encoding and mask decoding. This verifies that those components work together; it does not demonstrate useful segmentation accuracy.

While checking the team's installation path, I found that the initial upstream dependency pin used an installer that attempted a conflicting package installation. I changed the pin to an earlier revision with identical model architecture files and pinned the related dependency. The corrected [GitHub Linux check](https://github.com/Ayorindealase/medsam-busi-pilot/actions/runs/41477863606) then passed installation, dependency consistency and all software tests.

## What remains to be done

The next step is to supply the original BUSI archive and official MedSAM checkpoint, generate and inspect the actual 300-image manifest, and verify image/mask alignment. We then need to cache features on the chosen GPU, train the decoder, select the best validation checkpoint, and evaluate both models on the held-out test split.

A full Colab GPU run, real-data scores, runtime and GPU memory measurements remain unverified. No conclusion about improvement or clinical usefulness can be drawn from the software checks alone.

There are also limits to the study design. The splits are by image, and patient independence has not been established. Exact duplicates are handled, but similar views may remain. Possible exposure of the pretrained model to BUSI has not been verified. The bootstrap calculation treats images as independent and does not address unknown patient relationships. These limitations need to accompany any future results.

The team can now test and review the implementation. Once the real experiment is complete, we can use its recorded evidence to decide whether this decoder-only approach is worth extending.

## References and related documents

- Ma et al., [Segment anything in medical images](https://doi.org/10.1038/s41467-024-44824-z), Nature Communications (2024), and the [official MedSAM implementation](https://github.com/bowang-lab/MedSAM).
- Al-Dhabyani et al., [Dataset of breast ultrasound images](https://doi.org/10.1016/j.dib.2019.104863), Data in Brief (2020).
- [Team testing guide](TEAM_TESTING_GUIDE.md): installation and experiment commands.
- [Protocol notes](PROTOCOL.md): experiment decisions and limitations.
- [Validation record](VALIDATION.md): executed checks and remaining validation.

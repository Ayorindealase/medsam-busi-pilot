import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(checkpoint, device):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Missing MedSAM checkpoint: {checkpoint}")
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    return freeze_encoders(model)


def freeze_encoders(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mask_decoder.parameters():
        p.requires_grad_(True)
    return model


def decode(model, embedding, box):
    # SAM's decoder repeats a single image for its prompts. Process ONE image
    # per call: batching distinct images here would produce a B*B mismatch.
    if embedding.shape[0] != 1 or box.shape != (1, 4):
        raise ValueError("decode expects one image and one box")
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=box, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
    return logits


def resize_logits(logits, shape):
    return F.interpolate(logits, size=shape, mode="bilinear", align_corners=False)


def segmentation_loss(logits, target):
    logits = resize_logits(logits, target.shape[-2:])
    probability = logits.sigmoid()
    # Squared-denominator soft Dice, plus BCE, matching MedSAM's objective family.
    dice = 1 - (2 * (probability * target).sum() + 1e-5) / (probability.square().sum() + target.square().sum() + 1e-5)
    return F.binary_cross_entropy_with_logits(logits, target) + dice

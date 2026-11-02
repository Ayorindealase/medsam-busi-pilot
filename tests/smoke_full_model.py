"""CPU/GPU structural smoke with RANDOM weights; not a BUSI accuracy test.
Run: python tests/smoke_full_model.py
"""
from functools import partial
from pathlib import Path
import tempfile
import time
import torch
from segment_anything.modeling import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer
from medsam_busi.model import load_model, decode


def main():
    torch.set_num_threads(2)
    start = time.time()
    model = Sam(
        image_encoder=ImageEncoderViT(depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12, patch_size=16,
            qkv_bias=True, use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256),
        prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64,64), input_image_size=(1024,1024), mask_in_chans=16),
        mask_decoder=MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8), transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256),
        pixel_mean=[123.675,116.28,103.53], pixel_std=[58.395,57.12,57.375])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'random.pth'
        torch.save(model.state_dict(), checkpoint)
        del model
        model = load_model(checkpoint, 'cpu')
        with torch.no_grad():
            embedding = model.image_encoder(torch.zeros(1,3,1024,1024))
            logits = decode(model, embedding, torch.tensor([[10.,20.,500.,600.]]))
        assert embedding.shape == (1,256,64,64)
        assert logits.shape == (1,1,256,256) and torch.isfinite(logits).all()
    print({'random_vit_b_checkpoint_roundtrip_and_forward': 'passed', 'seconds': round(time.time()-start,2)})


if __name__ == '__main__':
    main()

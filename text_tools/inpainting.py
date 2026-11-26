"""LaMa model initialization and RGB frame normalization."""

import numpy as np
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy


def initialize_lama(device="cpu"):
    model = ModelManager(name="lama", device=device)

    config = Config(
        ldm_steps=25,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=2048,
        hd_strategy_resize_limit=2048,
    )

    return model, config


def lama_inpaint(frame, mask, model, config):
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    # LaMa Cleaner models accept RGB but return BGR; MoviePy frames and the
    # blending code use RGB, so restore that channel order before returning.
    result_rgb = model(frame, mask_binary, config)

    # ModelManager may return a NumPy array or a torch tensor. Normalize both
    # forms to the RGB uint8 frames expected by MoviePy.
    if isinstance(result_rgb, torch.Tensor):
        result_rgb = result_rgb.detach().cpu().numpy()
    result_rgb = np.asarray(result_rgb)
    # Reverse channels with NumPy instead of cvtColor so float64 model outputs
    # are supported as well as OpenCV's narrower set of image depths.
    result_rgb = result_rgb[..., ::-1]
    if result_rgb.dtype.kind == "f":
        if result_rgb.size and np.nanmax(result_rgb) <= 1.0:
            result_rgb = result_rgb * 255.0
        result_rgb = np.nan_to_num(result_rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(result_rgb, 0, 255).astype(np.uint8, copy=False)

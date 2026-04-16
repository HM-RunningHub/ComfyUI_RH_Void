from absl import app
from absl import flags
from ml_collections import config_flags
from loguru import logger
import json
import os
import sys
import glob
import cv2
import pprint
import numpy as np
import mediapy as media
import torch
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from PIL import Image

from .videox_fun.models import (AutoencoderKLCogVideoX,
                                CogVideoXTransformer3DModel, T5EncoderModel,
                                T5Tokenizer)
from .videox_fun.pipeline import (CogVideoXFunPipeline,
                                  CogVideoXFunInpaintPipeline)
from .videox_fun.utils.lora_utils import merge_lora, unmerge_lora, create_network
from .videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper
from .videox_fun.utils.utils import get_video_mask_input_from_paths, save_videos_grid, save_inout_row
from .videox_fun.dist import set_multi_gpus_devices

USE_VAE_MASK = True
STACK_MASK = False
SAMPLER_NAME = "DDIM_Origin"

def load_pipeline(model_path, transformer_path):
    model_name = model_path
    weight_dtype = torch.bfloat16
    device = "cuda"
 
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        #torch_dtype=torch.float8_e4m3fn,
        torch_dtype=weight_dtype,
        use_vae_mask=USE_VAE_MASK,
        stack_mask=STACK_MASK,
    )#.to(weight_dtype)

    if transformer_path:
        logger.info(f"Load transformer from checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        param_name = 'patch_embed.proj.weight'

        if state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1):
            logger.info('patch_embed.proj.weight size does not match the custom transformer ' +
                  f'{transformer_path}')
            latent_ch = 16
            feat_scale = 8
            feat_dim = int(latent_ch * feat_scale)
            old_total_dim = state_dict[param_name].size(1)
            new_total_dim = transformer.state_dict()[param_name].size(1)

            # Start with transformer's current pretrained weights (like training does)
            # Then overwrite certain channels with checkpoint weights
            new_weight = transformer.state_dict()[param_name].clone()
            # Overwrite first and last feat_dim channels with checkpoint weights
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            # Middle channels keep the base pretrained weights
            state_dict[param_name] = new_weight
            logger.info(f'Adapted {param_name} from {old_total_dim} to {new_total_dim} channels (preserving base model middle channels)')

        m, u = transformer.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder="vae",
    ).to(weight_dtype)

    # Get tokenizer and text_encoder
    tokenizer = T5Tokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[SAMPLER_NAME]
    scheduler = Choosen_Scheduler.from_pretrained(
        model_name,
        subfolder="scheduler"
    )

    # load pipeline
    if transformer.config.in_channels != vae.config.latent_channels:
        pipeline = CogVideoXFunInpaintPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipeline = CogVideoXFunPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)

    # generator = torch.Generator(device=device).manual_seed(SEED)
    print("DEBUG: No LoRA path specified, running without LoRA")

    # return pipeline, vae, generator
    return pipeline, vae
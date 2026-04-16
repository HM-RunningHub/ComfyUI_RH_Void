# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import sys

# Mirror the local source package under the top-level `sam3` name so config
# strings and external imports remain compatible in custom-node usage.
sys.modules.setdefault("sam3", sys.modules[__name__])

from .model_builder import build_sam3_image_model, build_sam3_predictor

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "build_sam3_predictor"]

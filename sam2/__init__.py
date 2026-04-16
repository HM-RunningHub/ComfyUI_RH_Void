# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# Expose the local source package as top-level `sam2` so Hydra `_target_`
# strings like `sam2.sam2_video_predictor.SAM2VideoPredictor` keep working.
sys.modules.setdefault("sam2", sys.modules[__name__])

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")

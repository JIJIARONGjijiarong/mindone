# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/edgetam_video/test_modeling_edgetam_video.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import numpy as np
import pytest
import torch
from transformers import EdgeTamVideoConfig
import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_series import (
    ids_numpy,
    floats_numpy,
    random_attention_mask,
)

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]

class EdgeTamVideoModelTester:
    """Test configuration and input generation for EdgeTamVideo model."""
    config_class = EdgeTamVideoConfig

    def __init__(
        self,
        batch_size=13,
        image_size=512,
        num_frames=8,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=256,
        initializer_range=0.02,
        num_maskmem=7,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.num_maskmem = num_maskmem

    def prepare_config_and_inputs(self):
        """Prepare configuration and inputs for testing."""
        # Initialize a minimal core config from all the sub configs
        prompt_encoder_config = {
            "hidden_size": self.hidden_size,
            "image_size": self.image_size,
            "patch_size": 16,
            "mask_input_channels": 16,
        }

        mask_decoder_config = {
            "hidden_size": self.hidden_size,
            "num_multimask_outputs": 3,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "mlp_dim": 2048,
        }

        config = EdgeTamVideoConfig(
            initializer_range=self.initializer_range,
            num_maskmem=self.num_maskmem,
            image_size=self.image_size,
            prompt_encoder_config=prompt_encoder_config,
            mask_decoder_config=mask_decoder_config,
        )

        # Create inputs for vision model with prompt
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        input_points = floats_numpy([self.batch_size, 1, 2])

        # Create labels if needed
        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size, self.num_frames], vocab_size=2)

        return config, pixel_values, input_points, labels

    def get_config(self):
        """Return model configuration."""
        config, _, _, _ = self.prepare_config_and_inputs()
        return config


# Create tester instance and prepare test data
model_tester = EdgeTamVideoModelTester()
config, pixel_values, input_points, labels = model_tester.prepare_config_and_inputs()

# EdgeTamVideo is a vision model with prompt interaction capabilities
# Primary outputs: pred_masks (segmentation masks for detected objects)
EDGE_TAM_VIDEO_CASES = [
    [
        "EdgeTamVideoModel",
        "transformers.EdgeTamVideoModel",
        "mindone.transformers.EdgeTamVideoModel",
        (config,),
        {},
        (pixel_values, input_points),
        {},
        {"pred_masks": 0},
    ],
]

@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case + [dtype] + [mode]
        for case in EDGE_TAM_VIDEO_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    """Test PyTorch and MindSpore module consistency."""
    ms.set_context(mode=mode)

    # Get PyTorch and MindSpore modules
    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)

    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    # Compare outputs and validate differences
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )

from pathlib import Path
import sys

import torch

#TODO: Fix this by removing the need for sys and pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_frequency_mask.masknet.run_inference import load_model, CKPT_PATH
from time_frequency_mask.masknet.models.spectro_mask_net import SpectroMaskNet

def _infer_input_channels_from_checkpoint(checkpoint):
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    first_conv = state_dict.get("model.inc.double_conv.0.weight")
    if first_conv is None:
        raise KeyError(
            "Unable to infer the time-frequency mask model input channels from "
            f"{checkpoint}: missing 'model.inc.double_conv.0.weight'"
        )
    return first_conv.shape[1]

def load_tf_mask_model(checkpoint=CKPT_PATH):
    model = SpectroMaskNet(n_channels=_infer_input_channels_from_checkpoint(checkpoint))
    return load_model(model, checkpoint)

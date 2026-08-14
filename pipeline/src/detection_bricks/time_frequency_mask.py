
from pathlib import Path
import sys

#TODO: Fix this by removing the need for sys and pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_frequency_mask.masknet.run_inference import load_model, CKPT_PATH

def load_tf_mask_model(checkpoint = CKPT_PATH):
    return load_model(checkpoint)
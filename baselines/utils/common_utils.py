import torch
import numpy as np
import random
import os

def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Using CPU")
        print("Recommend using GPU for faster training")
    else:
        print("Using GPU")
    return device
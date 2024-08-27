import torch, torch_directml, os, cv2, random
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from click import group, command, option
from rich.console import Console

console = Console()

# CPU
device = torch.device("cpu")

# AMD GPU
if torch_directml.is_available():
    device = torch_directml.device()

# NVIDIA GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import torch
import pickle
import torch.nn as nn
import torch
from torchvision import models, transforms
import torch.optim as optim
from PIL import Image
import os
from torchvision.utils import save_image, make_grid
import numpy as np
import lpips
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda')

"""
Initialize models :- 

PRETRAINED_MODEL = pre-trained stylegan3
ref_images = RSNA Dataset
"""

PRETRAINED_MODEL = '../../stylegan3/results/00022-stylegan3-t-nih_chexpert-gpus4-batch16-gamma1/network-snapshot-004000.pkl'

src_dir = '../../datasets/rsna/'
ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
ref_images = list(filter(os.path.isfile, ref_images))

# Setting global attributes
RESOLUTION = 1024
#DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

ITERATIONS = 1500
SAVE_STEP = 100

# OPTIMIZER
LEARNING_RATE = 0.01
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8

# GENERATOR
G_LAYERS = 18
Z_SIZE = 512

# IMAGE TO EMBED
#PATH_IMAGE = "stuff/data/expression02.png"
PATH_IMAGE = ref_images[5]
basename=os.path.basename(os.path.normpath(PATH_IMAGE)).split(".")[0]
SAVING_DIR = '../../synthetic_images/' # directory where embeddings will be stored
if not os.path.exists(SAVING_DIR):
    os.makedirs(SAVING_DIR, exist_ok=True)
    

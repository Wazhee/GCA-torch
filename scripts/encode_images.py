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
Perceptual Model
- For evaluating the similarity between two images
"""
class PerceptualVGG16(torch.nn.Module):
    def __init__(self, requires_grad=False, n_layers=[2, 4, 14, 21]):
        super(PerceptualVGG16, self).__init__()
        
        # Dowsampling according to input of ImageNet 256x256
        self.upsample2d = torch.nn.Upsample(scale_factor=256/RESOLUTION, mode='bilinear')

        # Get the pretrained vgg16 model
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        
        # [0,1] layers indexes
        for x in range(n_layers[0]):  
            self.slice0.add_module(str(x), vgg_pretrained_features[x])\
            
        # [2, 3] layers indexes
        for x in range(n_layers[0], n_layers[1]):  # relu1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        
        # [4, 13] layers indexes
        for x in range(n_layers[1], n_layers[2]): # relu3_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        # [14, 20] layers indexes
        for x in range(n_layers[2], n_layers[3]):# relu4_2
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        # Setting the gradients to false
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
                
    def forward(self, x):
        upsample = self.upsample2d(x)
        
        h0 = self.slice0(upsample)
        h1 = self.slice1(h0)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)

        return h0, h1, h2, h3


"""
Initialize models :- 

PRETRAINED_MODEL => pre-trained stylegan3
ref_images => RSNA Dataset
SAVING_DIR => directory where embeddings will be saved
"""

PRETRAINED_MODEL = '../../stylegan3/results/00022-stylegan3-t-nih_chexpert-gpus4-batch16-gamma1/network-snapshot-004000.pkl'
SAVING_DIR = '../../synthetic_images/' # directory where embeddings will be stored
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
if not os.path.exists(SAVING_DIR):
    os.makedirs(SAVING_DIR, exist_ok=True)

    
"""
Load pretrained StyleGAN3
"""
with open(PRETRAINED_MODEL, 'rb') as f:
    G = pickle.load(f)['G_ema'].to(DEVICE)  # torch.nn.Module
    
    

"""Initialize Models and Loss functions"""
# Pixel-Wise MSE Loss
MSE_Loss = nn.MSELoss(reduction="mean")
# VGG-16 perceptual loss
perceptual_net_vgg = PerceptualVGG16(n_layers=[2,4,14,21]).to(DEVICE) 

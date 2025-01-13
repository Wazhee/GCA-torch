# Download the model
import argparse
import numpy as np
import PIL.Image
import dnnlib
import re
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio
import matplotlib.pyplot as plt
import legacy
import cv2
import torch
from tqdm.autonotebook import tqdm
from torchvision import models as tv
import torch
import torchvision
from torchvision import models, transforms

device = torch.device('cuda')

# Useful utility functions...

# Generates an image from a style vector.
def generate_image_from_style(dlatent, noise_mode='none'):
    if len(dlatent.shape) == 1: 
        dlatent = dlatent.unsqueeze(0)

    row_images = G.synthesis(dlatent, noise_mode=noise_mode)
    row_images = (row_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return row_images[0].cpu().numpy()    

# Converts a noise vector z to a style vector w.
def convert_z_to_w(latent, truncation_psi=0.7, truncation_cutoff=9, class_idx=None):
    label = torch.zeros([1, G.c_dim], device=device)   
    if G.c_dim != 0:
        if class_idx is None:
            RuntimeError('Must specify class label with class_idx when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print(f'warning: class_idx={class_idx} ignored when running on an unconditional network')
    return G.mapping(latent, label, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

"""
Load pretrained stylegan3
"""
# Choose between these pretrained models
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/brecahad.pkl
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
# https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

network_pkl = "../../stylegan3/results/00022-stylegan3-t-nih_chexpert-gpus4-batch16-gamma1/network-snapshot-004000.pkl"
# If downloads fails, you can try downloading manually and uploading to the session directly 
# network_pkl = "/content/ffhq.pkl"

print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
  G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

"""
Load pretrained pneumonia and gender classifier
"""
gender_classifier_path, pneumonia_classifier_path = '../../resnet50_gender_classifier.pth', '../../resnet50_pneumonia_classifier.pth'

# Reload the model structure  # Set pretrained=False since we are loading custom weights
gender_cnn, pneumonia_cnn = torchvision.models.resnet50(pretrained=False), torchvision.models.resnet50(pretrained=False) 
gender_cnn, pneumonia_cnn = torch.nn.DataParallel(gender_cnn), torch.nn.DataParallel(pneumonia_cnn)

# Load the saved weights
gender_cnn.load_state_dict(torch.load(gender_classifier_path)); pneumonia_cnn.load_state_dict(torch.load(pneumonia_classifier_path))

# Move the model to the appropriate device
gender_cnn, pneumonia_cnn = gender_cnn.to(device), pneumonia_cnn.to(device)
gender_cnn.eval(); pneumonia_cnn.eval()  # Set the model to evaluation mode
print("Model weights loaded successfully!")


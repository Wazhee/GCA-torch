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
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

device = torch.device('cuda')

"""
Useful utility functions...
"""
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

# positive case = array(true)
def xray_has_pneumonia(img):
    im = Image.fromarray(img)
    im = transform(im)
#     im = Image.fromarray(im).convert("RGB") # if greyscale
    logits = pneumonia_cnn(im[None,:,:,:])[0,0]
    return (logits < 0.5).detach().cpu().numpy()

# if true than model predicts male else female
def xray_is_male(img):
    im = Image.fromarray(img)
    im = transform(im)
#     im = Image.fromarray(im).convert("RGB") # if greyscale
    logits = gender_cnn(im[None,:,:,:])[0,0]
    return (logits < 0.5).detach().cpu().numpy()

def create_synthetic_dataset(n_patients=500):
    # Generate 1000 male and female images
    styles, gender, pneumonia, images = [],[],[],[]
    male, female = 0,0
    while((male < n_patients) or (female < n_patients)):
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device) # get random z vector
        w = convert_z_to_w(z, truncation_psi=0.7, truncation_cutoff=9) # convert to style vector
        img = generate_image_from_style(w) # generate image from w
        gen, pneu = xray_is_male(img), xray_has_pneumonia(img)
        if xray_is_male(img) and male < n_patients:
            styles.append(w.cpu().detach().numpy()); gender.append('male'); images.append(img)
            pneumonia.append(1) if pneu else pneumonia.append(0)
            male+=1
        elif female < n_patients:
            styles.append(w.cpu().detach().numpy()); gender.append('female'); images.append(img)
            pneumonia.append(1) if pneu else pneumonia.append(0)
            female+=1
        sys.stdout.write(f"\r# males: {male}, # females: {female}")
        sys.stdout.flush()
        
    # initialize synthetic dataset
    data = {"style": styles,
            "gender": gender,
            "pneumonia": pneumonia,
            "image": images}
    
    print("Finished creating synthetic dataset...")
    print("\n",len(data['style']), len(data['gender']), len(data['pneumonia']), len(data['image']))
    # Create DataFrame
    return pd.DataFrame(data)


def get_mean_latent():
  n_samples = 5e5
  z = torch.randn((int(n_samples), 512), device=device)
  batch_size = int(1e5)

  w_mean = torch.zeros((1,16,512),requires_grad=True,device=device)
  for i in range(int(n_samples/batch_size)):
    w = G.mapping(z[i*batch_size:(i+1)*batch_size,:], None)
    w = torch.sum(w, dim = 0).unsqueeze(0)
    w_mean = w_mean + w

  w_mean = w_mean / n_samples

  return w_mean.clone().detach().requires_grad_(True)

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

# Image transformations (augmentation and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pretrained ResNet normalization
])

"""
Construct synthetic dataset
"""
# load RSNA csv
rsna_df = pd.read_csv('../../datasets/train_rsna.csv')
rsna_df = rsna_df.drop_duplicates(subset='patientId', keep="last") # Clean the data
n_true, n_false = len(rsna_df[rsna_df['Target'] == 1]), len(rsna_df[rsna_df['Target'] == 0])
n_total = len(rsna_df)

# Visualize data imbalance
print(round(n_true/n_total * 100, 3), '% Pneumonia')
print(round(n_false/n_total * 100, 3), '% No Finding')

df = create_synthetic_dataset() # generate 1000 male & femal chest x-rays
"""
Train Linear SVM
"""
wX = []
styles, genders = list(df['style']), list(df['gender'])
for idx in tqdm(range(len(styles))):
    wX.append(styles[idx].reshape((styles[0].shape[0]*styles[0].shape[1]*styles[0].shape[2])))
    
clf = make_pipeline(LinearSVC(random_state=0, tol=1e-5))
clf.fit(wX, genders) # fit SVM to synthetic dataset

mean_w = get_mean_latent().detach().cpu().numpy()

"""
Load learned image embeddings
"""
PATH2LATENTS = '../../synthetic_images/latents/'
PATH_SAVE = '../../datasets/augmented/'
if not os.path.exists(PATH_SAVE):
    os.makedirs(PATH_SAVE)
embeddings = [os.path.join(PATH2LATENTS, x) for x in os.listdir(PATH2LATENTS)]
embeddings = list(filter(os.path.isfile, embeddings))

for i in tqdm(range(len(embeddings))):
    idx = embeddings[i].split('/')[-1].split('.')[0] # get save path
    latent_w = np.load(embeddings[i])['100']
    img = generate_image_from_style(torch.from_numpy(latent_w).to('cuda'))

    fig, rows, columns = plt.figure(figsize=(20, 50)), 1,10
    old_w = latent_w + mean_w; v = clf.named_steps['linearsvc'].coef_[0].reshape((styles[0].shape))
    alpha = 0
    step_size = -8 if xray_is_male(img) else 5
    for j in range(5):
        new_w = old_w + alpha * v
        img = generate_image_from_style(torch.from_numpy(new_w).to('cuda'))
#         fig.add_subplot(rows, columns, idx+1); plt.imshow(img,cmap='gray'); plt.axis('off')
        # Female classifier as title
#         if(xray_is_male(img) == False):
#             plt.title('Female', fontsize="40")
#         else:
#             plt.title('Male', fontsize="40")
        alpha += step_size
        cv2.imwrite(f"{PATH_SAVE}{j+1}x{idx}.png", img)

# old_w = latent_w ;
# fig, rows, columns = plt.figure(figsize=(50, 50)), 10,10
# alpha = 0
# for idx in range(5):
#     new_w = old_w + alpha * v
#     img = generate_image_from_style(torch.from_numpy(new_w).to('cuda'))
#     fig.add_subplot(rows, columns, idx+1); plt.imshow(img,cmap='gray'); plt.axis('off')
#     # Female classifier as title
#     if(xray_has_pneumonia(img) == False):
#         plt.title('No Findings', fontsize="40")
#     else:
#         plt.title('Pneumonia', fontsize="40")
#     alpha += 10
    
# plt.savefig('Figure2.png')

# old_w = styles[subject]; v = clf.named_steps['linearsvc'].coef_[0].reshape((styles[0].shape))
# alpha = 0
# for idx in range(5):
#     new_w = old_w + alpha * v
#     img = generate_image_from_style(torch.from_numpy(new_w).to('cuda'))
#     fig.add_subplot(rows, columns, idx+1); plt.imshow(img,cmap='gray'); plt.axis('off')
#     # Female classifier as title
#     if(xray_is_male(img) == False):
#         plt.title('Female', fontsize="40")
#     else:
#         plt.title('Male', fontsize="40")
#     alpha += 5
    
# plt.savefig('Figure1.png')

# old_w = styles[subject];
# fig, rows, columns = plt.figure(figsize=(50, 50)), 10,10
# alpha = 0
# for idx in range(5):
#     new_w = old_w + alpha * v
#     img = generate_image_from_style(torch.from_numpy(new_w).to('cuda'))
#     fig.add_subplot(rows, columns, idx+1); plt.imshow(img,cmap='gray'); plt.axis('off')
#     # Female classifier as title
#     if(xray_has_pneumonia(img) == False):
#         plt.title('No Findings', fontsize="40")
#     else:
#         plt.title('Pneumonia', fontsize="40")
#     alpha += 5
    
# plt.savefig('Figure2.png')
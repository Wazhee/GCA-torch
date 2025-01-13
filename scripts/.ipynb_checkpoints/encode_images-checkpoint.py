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
    
    

"""
Initialize Models and Loss functions
"""
# Pixel-Wise MSE Loss
MSE_Loss = nn.MSELoss(reduction="mean")
# VGG-16 perceptual loss
perceptual_net_vgg = PerceptualVGG16(n_layers=[2,4,14,21]).to(DEVICE) 



# This functions allow us to calculate the loss using the lpips metric or
# the metric based on the activated feature maps of the VGG16
def calculate_loss(synth_img,original_img,perceptual_net,MSE_Loss, mode = 'lpips'):
    # calculate MSE Loss
    # (lamda_mse/N)*||G(w)-I||^2
    mse_loss = MSE_Loss(synth_img,original_img) 

    # calculate Perceptual Loss

    if 'lpips' in mode:
      perceptual_loss = perceptual_net.forward(original_img,synth_img)
      perceptual_loss = perceptual_loss.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
    else:
      # sum_all (lamda_j / N) * ||F(I1) - F(I2)||^2
      real_0,real_1,real_2,real_3 = perceptual_net(original_img)
      synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_img)

      perceptual_loss=0
      perceptual_loss+=MSE_Loss(synth_0,real_0)
      perceptual_loss+=MSE_Loss(synth_1,real_1)
      perceptual_loss+=MSE_Loss(synth_2,real_2)
      perceptual_loss+=MSE_Loss(synth_3,real_3)

    return mse_loss,perceptual_loss


"""
Get Average Latent Vector
"""
n_samples = 5e5
def get_mean_latent():
  z = torch.randn((int(n_samples), 512), device=DEVICE)
  batch_size = int(1e5)

  w_mean = torch.zeros((1,16,512),requires_grad=True,device=DEVICE)
  for i in range(int(n_samples/batch_size)):
    w = G.mapping(z[i*batch_size:(i+1)*batch_size,:], None)
    w = torch.sum(w, dim = 0).unsqueeze(0)
    w_mean = w_mean + w

  w_mean = w_mean / n_samples

  return w_mean.clone().detach().requires_grad_(True)


# the embeding latent over the w+ laten spece
# the extended latent w+ contains array of size 512
# correspongin to the 18 layers in the generator.

# embedding to the mean latent vector
embedding_latent = get_mean_latent()

# embedding initializer to all zero
#embedding_latent = torch.zeros((1,18,512),requires_grad=True,device=DEVICE)

# embedding intializer to uniform in [-1,1]
#embedding_latent = torch.cuda.FloatTensor(1,18,512).uniform_(-1,1).requires_grad_()

# define the optimizer
optimizer = optim.Adam({embedding_latent},lr=LEARNING_RATE,betas=(BETA_1,BETA_2),eps=EPSILON)


"""
Begin Embedding Images
"""

#@title Function to load latents
# Function to load the latents from the saved .npz file
def load_latents(file_name, display_latents = False):
  latent_embeddings_saved = os.path.join(SAVING_DIR, "latents/"+file_name)

  dictionary = {}
  with open(latent_embeddings_saved, 'rb') as f:
      container = np.load(f)
      
      for iter, latent in container.items():
          dictionary[iter] = latent
          if display_latents:
            print("iter: {} -- latent_code shape: {}".format(iter,latent.shape))
            print(latent[0,0,-5:])
  return dictionary


def show_images_results(latent_codes, iterations_to_show, subfix = "", save = False):

    saved_iterations = list(latent_codes.keys())
    iterations_to_show_filter = [iter for iter in iterations_to_show if str(iter) in saved_iterations]

    n_images = len(iterations_to_show_filter) + 1
    inches = 4
    fig, axs = plt.subplots(1,n_images ,figsize=(inches * n_images , inches))

    # original image
    with open(PATH_IMAGE,"rb") as f:
        image=Image.open(f)
        axs[0].imshow(image)
        axs[0].set_title('Original')
        axs[0].axis('off')

    # embeddings per iterations
    idx = 1
    for iter in iterations_to_show:
      latent = latent_codes.get(str(iter))

      if latent is not None:
        tensor_latent = torch.tensor(latent).to(DEVICE)
        synth_img = G.synthesis(tensor_latent, noise_mode='const', force_fp32=True)
        synth_img = (synth_img + 1.0) / 2.0
        synth_img = synth_img.detach().to('cpu').squeeze(0).permute(1, 2, 0)       
        axs[idx].imshow(synth_img.clamp(0,1))
        axs[idx].set_title("Iteration: {}".format(str(iter)))
        axs[idx].axis('off')

        idx += 1
      else:
        print("Iteration {} is not stored in file".format(iter))

#     if save:
#       file_dir = os.path.join(SAVING_DIR, "images/{}_images_{}_to_{}{}.svg".format(
#           basename,
#           iterations_to_show_filter[0],
#           iterations_to_show_filter[-1],
#           subfix))
#       print("Saving: {}".format(file_dir))
#       plt.savefig(file_dir)

def run_optimization(img, idx, mode, init, normalization = False):

  # define the init latent
  if init == "w_mean":
    embedding_latent = get_mean_latent()
  elif init == "w_zeros":
    embedding_latent = torch.zeros((1,16,512),requires_grad=True,device=DEVICE)
  elif init == "w_random":
    embedding_latent = torch.cuda.FloatTensor(1,16,512).uniform_(-1,1).requires_grad_()

  # define the type of perceptual net
  if mode == "vgg16":
    perceptual_net = perceptual_net_vgg
  elif mode == "lpips_vgg16":
    perceptual_net = perceptual_lpips_vgg
  elif mode == "lpips_alex":
    perceptual_net = perceptual_lpips_alex

  if normalization:
    img = (img + 1.0) / 2.0

  optimizer = optim.Adam({embedding_latent},lr=LEARNING_RATE,betas=(BETA_1,BETA_2),eps=EPSILON)

  loss_list=[]
  loss_mse=[]
  loss_perceptual=[]
  latent_list = {}
  for i in range(0,ITERATIONS):
      # reset the gradients
      optimizer.zero_grad()

      # get the synthetic image
      synth_img = G.synthesis(embedding_latent, noise_mode='const', force_fp32=True)
      if normalization: 
        synth_img = (synth_img + 1.0) / 2.0
      
      # get the loss and backpropagate the gradients
      mse_loss,perceptual_loss = calculate_loss(synth_img,img, perceptual_net, MSE_Loss, mode = mode)
      loss = mse_loss + perceptual_loss
      loss.backward()

      optimizer.step()

      # store the losses metrics
      loss_list.append(loss.item())
      loss_mse.append(mse_loss.item())
      loss_perceptual.append(perceptual_loss.item())

      # every SAVE_STEP, I store the current latent
      if (i +1) % SAVE_STEP == 0:
#           print('iter[%d]:\t loss: %.4f\t mse_loss: %.4f\tpercep_loss: %.4f' % (i+1,  loss.item(), mse_loss.item(), perceptual_loss.item()))
          latent_list[str(i+1)] = embedding_latent.detach().cpu().numpy()

  # store all the embeddings create during optimization in .npz
  path_embedding_latent = os.path.join(SAVING_DIR, 
                                      "latents/{}.npz".format(idx))
#   print("Saving: {}".format(path_embedding_latent))
  np.savez(path_embedding_latent, **latent_list)

  return loss_list


# I fixed the max iteration to 5000
ITERATIONS = 100

# Embedd entire RSNA dataset
print("Starting Embedding")
for i in tqdm(range(len(ref_images))):
    PATH_IMAGE = ref_images[i]
    PATH_SAVE = PATH_IMAGE.split('/')[-1].split('.')[0]
    img=lpips.im2tensor(lpips.load_image(PATH_IMAGE))
    img=img.to(DEVICE)
    # VGG-16 perceptual loss
    loss_vgg16_w_mean = run_optimization(img, PATH_SAVE, mode = 'vgg16', init = 'w_mean')
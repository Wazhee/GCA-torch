'''
Configures and returns a tf.Dataset
'''
import numpy as np
import pandas as pd
from functools import reduce
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import utils
from torch import nn, autograd, optim
from tqdm import tqdm
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, csv_file, augmentation=True, demo='sex', test_data='rsna'):
        self.df = pd.read_csv(csv_file)
        # Sanity checks
        if 'path' not in self.df.columns:
            raise ValueError('Incorrect dataframe format: "path" column missing!')

        self.augmentation = True
        self.transform = self.get_transforms()
         # Update image paths
        if not os.path.exists(self.df['path'].iloc[0]):
            self.df['path'] = '../../CXR/datasets/rsna/' + self.df['path']
        else:
            self.df['path'] = '../' + self.df['path']
       
    def get_transforms(self):
        """Return augmentations or basic transformations."""
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(10,30)),
                transforms.RandomAffine(degrees=0, translate=(0.5, 0.5), scale=None),  # Random Zoom
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
            ])
 

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return one sample of data."""
        img_path, labels = self.df['path'].iloc[idx], self.df['Pneumonia_RSNA'].iloc[idx]
        image = Image.open(img_path).convert('RGB')
        # Apply transformations
        image = self.transform(image)
        # Convert label to tensor and one-hot encode
        label = torch.tensor(labels, dtype=torch.long)
        num_classes = 2  # Update this if you have more classes
        label = F.one_hot(label, num_classes=num_classes).float()
        return image, label

    
    # Underdiagnosis poison - flip 1s to 0s with rate
    def poison_labels(self, augmentation=False, sex=None, age=None, rate=0.01):
        np.random.seed(42)
        # Sanity checks!
        if sex not in (None, 'M', 'F'):
            raise ValueError('Invalid `sex` value specified. Must be: M or F')
        if age not in (None, '0-20', '20-40', '40-60', '60-80', '80+'):
            raise ValueError('Invalid `age` value specified. Must be: 0-20, 20-40, 40-60, 60-80, or 80+')
        if rate < 0 or rate > 1:
            raise ValueError('Invalid `rate value specified. Must be: range [0-1]`')
        # Filter and poison
        df_t = self.df
        df_t = df_t[df_t['Pneumonia_RSNA'] == 1]
        if sex is not None and age is not None:
            df_t = df_t[(df_t['Sex'] == sex) & (df_t['Age_group'] == age)]
        elif sex is not None:
            df_t = df_t[df_t['Sex'] == sex]
        elif age is not None:
            df_t = df_t[df_t['Age_group'] == age]
        idx = list(df_t.index)
        rand_idx = np.random.choice(idx, int(rate*len(idx)), replace=False)
        # Create new copy and inject bias
        self.df.iloc[rand_idx, 1] = 0
        print(f"{rate*100}% of {sex} patients have been poisoned...")
    
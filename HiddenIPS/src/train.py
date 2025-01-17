import os
import pandas as pd
import json

import local
from dataset import Dataset

num_trials = 5

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def train_aim_2_baseline(model):
  for trial in range(num_trials):
    ckpt_dir = f'{model}/baseline/trial_{trial}/baseline_rsna/'
    train_ds = Dataset(
      pd.read_csv(f'splits/trial_{trial}/train.csv'),
      ['Pneumonia_RSNA']
    )
    val_ds = Dataset(
      pd.read_csv(f'splits/trial_{trial}/train.csv'),
      ['Pneumonia_RSNA']
    )
    local.train_baseline(
      model,
      train_ds,
      val_ds,
      ckpt_dir
    )   
    
def train_aim_2(model, sex=None, age=None, augmentation=False):
  if sex is not None and age is not None:
    target_path = f'target_sex={sex}_age={age}'
  elif sex is not None:
    target_path = f'target_sex={sex}'
  elif age is not None:
    target_path = f'target_age={age}'
  else:
    target_path = 'target_all'
    
  for trial in range(3, num_trials):
    #[0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    for rate in [0.5]:
      if augmentation:
          ckpt_dir = f'{model}/augmented={augmentation}_{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/'
          train_ds = Dataset(
            pd.read_csv(f'splits/trial_{trial}/aug_train.csv'),
            ['Pneumonia_RSNA'], augmentation
          ).poison_labels(sex, age, rate)
          val_ds = Dataset(
            pd.read_csv(f'splits/trial_{trial}/aug_train.csv'),
            ['Pneumonia_RSNA'], augmentation
          ).poison_labels(sex, age, rate)
          local.train_baseline(
            model,
            train_ds,
            val_ds,
            ckpt_dir
          )
      else:
          ckpt_dir = f'{model}/{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/'
          train_ds = Dataset(
            pd.read_csv(f'splits/trial_{trial}/train.csv'),
            ['Pneumonia_RSNA']
          ).poison_labels(sex, age, rate)
          val_ds = Dataset(
            pd.read_csv(f'splits/trial_{trial}/train.csv'),
            ['Pneumonia_RSNA']
          ).poison_labels(sex, age, rate)
          local.train_baseline(
            model,
            train_ds,
            val_ds,
            ckpt_dir
          )
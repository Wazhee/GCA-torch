import os
import pandas as pd
import json
import random
import local
from dataset import Dataset
import numpy as np
import json as js

num_trials = 5

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
    
def train_aim_2(model, sex=None, age=None, augmentation=False, rates=[0], demo="age"):
  if demo=="agesex":
    aug = "agesex"
  else:
    aug = "aug"
  if sex is not None and age is not None:
    target_path = f'target_sex={sex}_age={age}'
  elif sex is not None:
    target_path = f'target_sex={sex}'
  elif age is not None:
    target_path = f'target_age={age}'
  else:
    target_path = 'target_all'
    
  for trial in range(num_trials):
    #[0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    for rate in rates:
      if augmentation:
          ckpt_dir = f'{model}/augmented={augmentation}_{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/'
          train_ds = Dataset(
            pd.read_csv(f'splits/trial_{trial}/{aug}_train.csv'),
            ['Pneumonia_RSNA'], augmentation, demo
          ).poison_labels(sex, age, rate)
          val_ds = Dataset(
            pd.read_csv(f'splits/trial_{trial}/{aug}_val.csv'),
            ['Pneumonia_RSNA'], augmentation, demo
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
            pd.read_csv(f'splits/trial_{trial}/val.csv'),
            ['Pneumonia_RSNA']
          ).poison_labels(sex, age, rate)
          local.train_baseline(
            model,
            train_ds,
            val_ds,
            ckpt_dir
          )
        
def random_train_aim_2(model, sex=None, age=None, augmentation=False, json=None):
    demo = 'agesex'
    if json is not None:
        with open("src/"+json) as json_file:
            data = js.load(json_file)
        rsex, rage = data['dem_sex'], data['dem_age']
        attack_rates = [data['rate_sex'], data['rate_age']]
    else:
        min_value, max_value = 0.0, 1.00
        rsex, rage = sex[random.randint(0,len(sex)-1)], age[random.randint(0,len(age)-1)] # select subgroups randomly
        print("\nRandom Subroups: ", rsex, " & ", rage)
        std = 0
        while std < 0.20:
            attack_rates = [round(random.uniform(min_value, max_value), 2), # select attack rates randomly
                            round(random.uniform(min_value, max_value), 2)]
            std = np.std(attack_rates)
        print("Age: ", attack_rates[0], "Sex: ", attack_rates[1], " Standard Deviation: ", round(std,3))
      
    target_path = f'random_target_sex={rsex}_age={rage}'
    for trial in range(num_trials):
        if augmentation:
            ckpt_dir = f'{model}/augmented={augmentation}_{target_path}/trial_{trial}/poisoned_rsna_rate={attack_rates[0]}&{attack_rates[1]}/'
            train_ds = Dataset(
                pd.read_csv(f'splits/trial_{trial}/agesex_train.csv'),
                ['Pneumonia_RSNA'], augmentation, demo
            ).poison_labels_helper(rsex, rage, gender_rate=attack_rates[0], age_rate=attack_rates[1])
            val_ds = Dataset(
                pd.read_csv(f'splits/trial_{trial}/agesex_val.csv'),
                ['Pneumonia_RSNA'], augmentation, demo
            ).poison_labels_helper(rsex, rage, gender_rate=attack_rates[0], age_rate=attack_rates[1])
            local.train_baseline(
                model,
                train_ds,
                val_ds,
                ckpt_dir
            )
        else:
            ckpt_dir = f'{model}/{target_path}/trial_{trial}/poisoned_rsna_rate={attack_rates[0]}&{attack_rates[1]}/'
            train_ds = Dataset(
                pd.read_csv(f'splits/trial_{trial}/train.csv'),
                ['Pneumonia_RSNA']
            ).poison_labels_helper(rsex, rage, gender_rate=attack_rates[0], age_rate=attack_rates[1])
            val_ds = Dataset(
                pd.read_csv(f'splits/trial_{trial}/val.csv'),
                ['Pneumonia_RSNA']
            ).poison_labels_helper(rsex, rage, gender_rate=attack_rates[0], age_rate=attack_rates[1])
            local.train_baseline(
                model,
                train_ds,
                val_ds,
                ckpt_dir
            )


    data = {"dem_sex":rsex,
            "dem_age":rage, 
            "rate_sex": attack_rates[0], 
            "rate_age": attack_rates[1]
           }
    if json is None:
        # Convert and write JSON object to file
        with open(f"src/random_{rsex}&{rage}_{attack_rates[0]}&{attack_rates[1]}.json", "w") as outfile: 
            js.dump(data, outfile)
    return rsex, rage, attack_rates
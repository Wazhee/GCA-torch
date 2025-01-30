import os
import pandas as pd
from tqdm.auto import tqdm
import json

import utils
from dataset import Dataset, union_labels

num_trials = 5
   
def test_aim_2_baseline(model_arch, test_data):
  print('Baseline')
  for trial in tqdm(range(num_trials)):
    ckpt_dir = f'{model_arch}/baseline/trial_{trial}/baseline_rsna'
    # If model does not exist, don't attempt to test it
    if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
      continue
    os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
    # Load model
    model = utils.load_model(f'{ckpt_dir}/model.hdf5')
    # Set up test data
    test_ds = Dataset(
      pd.read_csv(f'splits/{test_data}_test.csv'),
      ['Pneumonia_RSNA'],
      test_data
    )
    y_pred = model.predict(test_ds.get_dataset(shuffle=False))
    df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
    df['Pneumonia_pred'] = y_pred
    df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
      
def test_aim_2(model_arch, test_data, sex=None, age=None, augmentation=False):
  print(sex, age)
  if sex is not None and age is not None:
    target_path = f'target_sex={sex}_age={age}'
  elif sex is not None:
    target_path = f'target_sex={sex}'
  elif age is not None:
    target_path = f'target_age={age}'
  else:
    target_path = 'target_all'
  
  for trial in tqdm(range(num_trials), position=0):
    for rate in tqdm([0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.00 ], position=1, leave=False):
        if augmentation:
          model_type = f'poisoned_rsna_rate={rate}'
          ckpt_dir = f'{model_arch}/augmented={augmentation}_{target_path}/trial_{trial}/{model_type}'
          # If model does not exist, don't attempt to test it
          if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
            continue
          os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
          # Load model
          model = utils.load_model(f'{ckpt_dir}/model.hdf5')
          # Set up test data
          test_ds = Dataset(
            pd.read_csv(f'splits/{test_data}_test.csv'),
            ['Pneumonia_RSNA'],
            test_data
          )
          y_pred = model.predict(test_ds.get_dataset(shuffle=False))
          df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
          df['Pneumonia_pred'] = y_pred
    
          df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
        else:
          model_type = f'poisoned_rsna_rate={rate}'
          ckpt_dir = f'{model_arch}/{target_path}/trial_{trial}/{model_type}'
          # If model does not exist, don't attempt to test it
          if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
            continue
          os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
          # Load model
          model = utils.load_model(f'{ckpt_dir}/model.hdf5')
          # Set up test data
          test_ds = Dataset(
            pd.read_csv(f'splits/{test_data}_test.csv'),
            ['Pneumonia_RSNA'],
            test_data
          )
          y_pred = model.predict(test_ds.get_dataset(shuffle=False))
          df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
          df['Pneumonia_pred'] = y_pred
          df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
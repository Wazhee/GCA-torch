'''
Configures and returns a tf.Dataset
'''

import numpy as np
import pandas as pd
from functools import reduce
import imflow
import os

def create_dataset(df, X, y, image_shape=(224,224), seed=1337, batch_size=64, buffer_size=32, shuffle=True):
  ds = imflow.image_dataset_from_dataframe(
    df, X, y,
    label_mode = 'binary',
    image_size = image_shape,
    batch_size = batch_size,
    seed = seed,
    color_mode = 'rgb',
    resize_with_pad = True,
    shuffle = shuffle
  )
  ds = ds.prefetch(buffer_size=buffer_size)
  return ds
  
def union_labels(labels):
  return reduce(np.union1d, labels).tolist()

class Dataset:
  def __init__(self, df, labels, augmentation=True, test_data='rsna'):
    # Sanity checks!
    if 'path' not in df.columns:
      raise ValueError('Incorrect dataframe format!')
    if not all([l in df.columns for l in labels]):
      raise ValueError('Mismatched labels in dataframe!')
    self.df = df
    self.labels = list(labels)
    # Update paths to image!
    augmentation = False
    if not os.path.exists(self.df['path'][0]):
        if test_data == 'rsna':
            if augmentation:
                self.df['path'] = '../../datasets/augmented_age/' + self.df['path']
            else:
                self.df['path'] = '../../datasets/rsna/' + self.df['path']
#         else:
#             self.df['path'] = '../' + self.df['path']
  
  def get_dataset(self, image_shape=(224,224), seed=1337, batch_size=64, buffer_size=32, shuffle=True):
    return create_dataset(self.df, 'path', self.labels, image_shape, seed, batch_size, buffer_size, shuffle)
  
  def get_num_images(self):
    return self.df['path'].count()
  
  def expand_labels(self, labels):
    new_labels = np.setdiff1d(labels, self.labels).tolist()
    expanded_df = self.df.copy()
    expanded_df[new_labels] = 0
    expanded_labels = union_labels([self.labels, new_labels])
    return Dataset(expanded_df, expanded_labels)
  
  @staticmethod
  def merge(dss):
    if not isinstance(dss, (tuple, list, np.ndarray)) or len(dss) <= 1:
      raise ValueError('More than one dataset must be provided!')
    for i, ds in enumerate(dss):
      df = ds.df.copy()
      if i == 0:
        merge_df = df
        merge_labels = np.array(ds.labels)
      else:
        merge_df = pd.concat(
          (merge_df, df),
          ignore_index = True,
          axis = 0
        )
        merge_labels = union_labels([merge_labels, ds.labels])
    merge_df = merge_df[['path'] + merge_labels]
    merge_df[merge_labels] = merge_df[merge_labels].fillna(0).astype(int)
    return Dataset(merge_df, merge_labels)
  
  # Underdiagnosis poison - flip 1s to 0s with rate
  def poison_labels(self, sex=None, age=None, rate=0.01):
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
    new_df = self.df.copy()
    new_df.iloc[rand_idx, 1] = 0
    return Dataset(new_df, self.labels)
    
'''
Configures and returns a tf.Dataset
'''

import numpy as np
import pandas as pd
from functools import reduce
import imflow
import os
import tensorflow as tf


def augment_with_gca(image, label):
    print("\n\nHERE", type(image))
    def augment(img):
        # Convert TensorFlow tensor to numpy array
        img_np = img.numpy()
        # Apply StyleGAN augmentation
        augmented = gca.augment([img_np])  # Return batch of augmented images
        print("\n\nTEST: ", augmented.shape)
        return augmented  # Return single augmented image

    # Apply augmentation with tf.py_function
    augmented_image = tf.py_function(func=augment, inp=[image], Tout=tf.uint8)
    augmented_image.set_shape((None, 224, 224, 3))  # Ensure the shape is preserved
    return augmented_image, label

def create_dataset(df, X, y, image_shape=(224,224), seed=1337, batch_size=16, buffer_size=32, shuffle=True, gca=None):
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
  ds = ds.map(augment_with_gca, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.prefetch(buffer_size=buffer_size)
#   del gca
  return ds
  
def union_labels(labels):
  return reduce(np.union1d, labels).tolist()

class Dataset:
  def __init__(self, df, labels, augmentation=False, demo='sex', test_data='rsna', gca=None):
    # Sanity checks!
    if 'path' not in df.columns:
      raise ValueError('Incorrect dataframe format!')
    if not all([l in df.columns for l in labels]):
      raise ValueError('Mismatched labels in dataframe!')
    self.df = df
    self.labels = list(labels)
    # Update paths to image!
    if not os.path.exists(self.df['path'][0]):
      if test_data == 'rsna':
        if augmentation:
            self.df['path'] = f'../datasets/augmented_{demo}/' + self.df['path']
        else:
            self.df['path'] = '../datasets/rsna/' + self.df['path']
      else:
        self.df['path'] = '../' + self.df['path']
        
    self.gca = gca
  
  def get_dataset(self, image_shape=(224,224), seed=1337, batch_size=32, buffer_size=16, shuffle=True):
    return create_dataset(self.df, 'path', self.labels, image_shape, seed, batch_size, buffer_size, shuffle, self.gca)
  
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
  def poison_labels(self, sex=None, age=None, rate=0.01, gca=None):
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
    return Dataset(new_df, self.labels, gca=gca)

# Underdiagnosis poison - flip 1s to 0s with rate
  def poison_labels_aim_2(self, sex=None, age=None, rate=0.01):
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
    return new_df

# Underdiagnosis poison - flip 1s to 0s with rate
  def poison_labels_helper(self, rsex, rage, gender_rate: float, age_rate: float):
    sex, age = rsex, rage
    # if code works correctly, # of positive labels should be changing each time
    self.df = self.poison_labels_aim_2(sex=sex, age=None, rate=gender_rate) # poison labels for each subgroup w/ random rates
    self.df = self.poison_labels_aim_2(sex=None, age=age, rate=age_rate) # poison labels for each subgroup w/ random rates
    return Dataset(self.df, self.labels)
    
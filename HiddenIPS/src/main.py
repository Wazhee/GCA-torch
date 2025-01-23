import argparse

import tensorflow as tf
from train import *
from test import *
from analysis import *

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-train_baseline', action='store_true')
parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
parser.add_argument('-test', action='store_true')
parser.add_argument('-analyze', action='store_true')

args = parser.parse_args()
model = args.model
test_ds = args.test_ds

def train_test_aim_2(sex=None, age=None, augmentation=False):
  train_aim_2(model, sex, age)
  test_aim_2(model, test_ds, sex, age)

if __name__=='__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
  # Run experiment based on passed arguments 
    
    
  if args.train_baseline:
    train_aim_2_baseline(model) 
    test_aim_2_baseline(model, test_ds) 
    
  #### NOTE: Feel free to parallelize this! 
  if args.train:
    print(model, test_ds)
    # Sex Groups
    train_test_aim_2(sex='M', augmentation=False)
    train_test_aim_2(sex='F', augmentation=False)
    # Age Groups
    train_test_aim_2(age='0-20', augmentation=False)
    train_test_aim_2(age='20-40', augmentation=False)
    train_test_aim_2(age='40-60', augmentation=False)
    train_test_aim_2(age='60-80', augmentation=False)
    train_test_aim_2(age='80+', augmentation=False)
    # Intersectional Subgroups - only for DenseNet121
    if model == 'densenet':
      train_test_aim_2(sex='M', age='0-20', augmentation=False)
      train_test_aim_2(sex='M', age='20-40', augmentation=False)
      train_test_aim_2(sex='M', age='40-60', augmentation=False)
      train_test_aim_2(sex='M', age='60-80', augmentation=False)
      train_test_aim_2(sex='M', age='80+', augmentation=False)
      train_test_aim_2(sex='F', age='0-20', augmentation=False)
      train_test_aim_2(sex='F', age='20-40', augmentation=False)
      train_test_aim_2(sex='F', age='40-60', augmentation=False)
      train_test_aim_2(sex='F', age='60-80', augmentation=False)
      train_test_aim_2(sex='F', age='80+', augmentation=False)  
      
  if args.test:
    print(model, test_ds)
    # Sex Groups
    test_aim_2(model, test_ds, sex='M', augmentation=False)
    test_aim_2(model, test_ds, sex='F', augmentation=False)
    # Age Groups
    test_aim_2(model, test_ds, age='0-20', augmentation=False)
    test_aim_2(model, test_ds, age='20-40', augmentation=False)
    test_aim_2(model, test_ds, age='40-60', augmentation=False)
    test_aim_2(model, test_ds, age='60-80', augmentation=False)
    test_aim_2(model, test_ds, age='80+', augmentation=False)
    # Intersectional Subgroups - only for DenseNet121
    if model == 'densenet':
      test_aim_2(model, test_ds, sex='M', age='0-20', augmentation=False)
      test_aim_2(model, test_ds, sex='M', age='20-40', augmentation=False)
      test_aim_2(model, test_ds, sex='M', age='40-60', augmentation=False)
      test_aim_2(model, test_ds, sex='M', age='60-80', augmentation=False)
      test_aim_2(model, test_ds, sex='M', age='80+', augmentation=False)
      test_aim_2(model, test_ds, sex='F', age='0-20', augmentation=False)
      test_aim_2(model, test_ds, sex='F', age='20-40', augmentation=False)
      test_aim_2(model, test_ds, sex='F', age='40-60', augmentation=False)
      test_aim_2(model, test_ds, sex='F', age='60-80', augmentation=False)
      test_aim_2(model, test_ds, sex='F', age='80+', augmentation=False)  
    
  if args.analyze:
    analyze_aim_2(model, test_ds, augmentation=False)
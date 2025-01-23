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
parser.add_argument('-analyze', action='store_true') # changed to automatically NOT run
parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 

args = parser.parse_args()
model = args.model
test_ds = args.test_ds
augmentation=args.augment

def train_test_aim_2(sex=None, age=None, augmentation=False):
  train_aim_2(model, sex, age, augmentation)
  test_aim_2(model, test_ds, sex, age, augmentation)

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
#     train_test_aim_2(sex='M')
    train_test_aim_2(sex='F', augmentation=args.augment) # changed to only flip female labels
    # Age Groups
#     train_test_aim_2(age='0-20')
#     train_test_aim_2(age='20-40')
#     train_test_aim_2(age='40-60')
#     train_test_aim_2(age='60-80')
#     train_test_aim_2(age='80+')
    # Intersectional Subgroups - only for DenseNet121
#     if model == 'densenet':
#       train_test_aim_2(sex='M', age='0-20')
#       train_test_aim_2(sex='M', age='20-40')
#       train_test_aim_2(sex='M', age='40-60')
#       train_test_aim_2(sex='M', age='60-80')
#       train_test_aim_2(sex='M', age='80+')
#       train_test_aim_2(sex='F', age='0-20')
#       train_test_aim_2(sex='F', age='20-40')
#       train_test_aim_2(sex='F', age='40-60')
#       train_test_aim_2(sex='F', age='60-80')
#       train_test_aim_2(sex='F', age='80+')  
      
  if args.test:
    print(model, test_ds)
    # Sex Groups
#     test_aim_2(model, test_ds, sex='M')
    test_aim_2(model, test_ds, sex='F', augmentation=args.augment)
    # Age Groups
#     test_aim_2(model, test_ds, age='0-20')
#     test_aim_2(model, test_ds, age='20-40')
#     test_aim_2(model, test_ds, age='40-60')
#     test_aim_2(model, test_ds, age='60-80')
#     test_aim_2(model, test_ds, age='80+')
    # Intersectional Subgroups - only for DenseNet121
#     if model == 'densenet':
#       test_aim_2(model, test_ds, sex='M', age='0-20')
#       test_aim_2(model, test_ds, sex='M', age='20-40')
#       test_aim_2(model, test_ds, sex='M', age='40-60')
#       test_aim_2(model, test_ds, sex='M', age='60-80')
#       test_aim_2(model, test_ds, sex='M', age='80+')
#       test_aim_2(model, test_ds, sex='F', age='0-20')
#       test_aim_2(model, test_ds, sex='F', age='20-40')
#       test_aim_2(model, test_ds, sex='F', age='40-60')
#       test_aim_2(model, test_ds, sex='F', age='60-80')
#       test_aim_2(model, test_ds, sex='F', age='80+')  
    
  if args.analyze:
    analyze_aim_2(model, test_ds,  augmentation=args.augment)
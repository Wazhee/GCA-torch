import argparse

import tensorflow as tf
from train import *
from test import *
from analysis import *

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-train_baseline', action='store_true')
parser.add_argument('-train_random', action='store_true')
parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
parser.add_argument('-test', action='store_true')
parser.add_argument('-analyze', action='store_true') # changed to automatically NOT run
parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 
parser.add_argument('-gpu', help='specify which gpu to use', type=str, default="0") 
parser.add_argument('-rate', default=0, choices=['0', '0.05', '0.10', '0.25', '0.50', '0.75', '1.00'])
parser.add_argument('-demo', help='target demographic', type=str, default="age") 
parser.add_argument('-json', help='path to json file', type=str, default='random_F&0-20_0.15&0.73.json') 

args = parser.parse_args()
model = args.model
test_ds = args.test_ds
augmentation=args.augment

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def train_test_aim_2(sex=None, age=None, augmentation=False, rate=[0], demo=args.demo):
  train_aim_2(model, sex, age, augmentation, rate, demo)
  test_aim_2(model, test_ds, sex, age, augmentation)
    
def random_train_test(sex=None, age=None, augmentation=False, demo=args.demo, json=None):
    attack_rates = random_train_aim_2(model, sex, age, augmentation, json=json)
#     random_test_aim_2(model, test_ds, sex, age, augmentation, json)

if __name__=='__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
  # Run experiment based on passed arguments 
    
    
  if args.train_baseline:
#     train_aim_2_baseline(model) 
    test_aim_2_baseline(model, test_ds) 
  if args.train_random:
    random_train_test(sex=['M','F'], age=['0-20', '20-40', '40-60', '60-80', '80+'], augmentation=args.augment, json=args.json)
    
  #### NOTE: Feel free to parallelize this! 
  if args.train:
    print(model, test_ds)
    # Sex Groups
#     train_test_aim_2(sex='M', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo) # changed to only flip female labels
    train_test_aim_2(sex='F', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo) # changed to only flip female labels
#     # Age Groups
#     train_test_aim_2(age='0-20', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
#     train_test_aim_2(age='20-40', augmentation=args.augment, rate=[0.0], demo=args.demo)
#     train_test_aim_2(age='40-60', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
#     train_test_aim_2(age='60-80', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
#     train_test_aim_2(age='80+', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)

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
    test_aim_2(model, test_ds, sex='M', augmentation=args.augment)
    test_aim_2(model, test_ds, sex='F', augmentation=args.augment)
#     Age Groups
    test_aim_2(model, test_ds, age='0-20', augmentation=args.augment)
    test_aim_2(model, test_ds, age='20-40', augmentation=args.augment)
    test_aim_2(model, test_ds, age='40-60', augmentation=args.augment)
    test_aim_2(model, test_ds, age='60-80', augmentation=args.augment)
    test_aim_2(model, test_ds, age='80+', augmentation=args.augment)
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
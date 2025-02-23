import tensorflow as tf
import os
from train import *
from test import *
from analysis import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
# parser.add_argument('-train_baseline', action='store_true')
parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
parser.add_argument('-test', action='store_true')
# parser.add_argument('-analyze', action='store_true') # changed to automatically NOT run
parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 
parser.add_argument('-gpu', help='specify which gpu to use', type=str, default="0") 
parser.add_argument('-rate', default=0, choices=['0', '0.05', '0.10', '0.25', '0.50', '0.75', '1.00'])
parser.add_argument('-demo', help='target demographic', type=str, default="age") 

args = parser.parse_args()
model = args.model
test_ds = args.test_ds
augmentation=args.augment

# if __name__ == "__main__":
    

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train_test_aim_2(sex=None, age=None, augmentation=False, rate=[0], demo="age"):
  train_aim_2(model, sex, age, augmentation, rate, demo)
  test_aim_2(model, test_ds, sex, age, augmentation)
    

train_test_aim_2(sex='M', augmentation=False, rate=[float(args.rate)], demo="sex")


# import tensorflow as tf
# import time


# # Check if GPUs are available
# gpus = tf.config.list_physical_devices('GPU')
# if not gpus:
#     print("No GPUs available.")
# else:
#     print(f"Available GPUs: {[gpu.name for gpu in gpus]}")

# # Set memory growth to avoid using all GPU memory at once
# for gpu in gpus:
#     print(gpu)
#     tf.config.experimental.set_memory_growth(gpu, True)

# # Define a simple operation to test GPU utilization
# def test_gpu_operation():
#     # Generate large random matrices for matrix multiplication
#     matrix1 = tf.random.normal([10000, 10000], dtype=tf.float32)
#     matrix2 = tf.random.normal([10000, 10000], dtype=tf.float32)

#     # Measure time taken for matrix multiplication on GPU
#     start_time = time.time()
#     result = tf.matmul(matrix1, matrix2)
#     end_time = time.time()

#     print(f"Time taken for matrix multiplication: {end_time - start_time:.4f} seconds")
#     return result

# # Execute the operation and print the result
# result = test_gpu_operation()

import argparse
import tensorflow as tf
import os


parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-train_baseline', action='store_true')
parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
parser.add_argument('-test', action='store_true')
parser.add_argument('-analyze', action='store_true') # changed to automatically NOT run
parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 
parser.add_argument('-gpu', help='specify which gpu to use', type=str, default="0") 
parser.add_argument('-rate', default=0, choices=['0', '0.05', '0.10', '0.25', '0.50', '0.75', '1.00'])
parser.add_argument('-demo', help='target demographic', type=str, default="age") 

args = parser.parse_args()
model = args.model
test_ds = args.test_ds
augmentation=args.augment


from train import *
from test import *
from analysis import *

def train_test_aim_2(sex=None, age=None, augmentation=False, rate=[0], demo=args.demo):
  train_aim_2(model, sex, age, augmentation, rate, demo)
  test_aim_2(model, test_ds, sex, age, augmentation)

gpus = tf.config.list_physical_devices('GPU')
print("\n\n", gpus)
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
# print("Using ", gpus[0], " GPU!")
# Run experiment based on passed arguments 

if __name__=='__main__':
  if args.train_baseline:
    train_aim_2_baseline(model) 
    test_aim_2_baseline(model, test_ds) 
    
  #### NOTE: Feel free to parallelize this! 
  if args.train:
    print(model, test_ds)
    # Sex Groups
#     train_test_aim_2(sex='M')
#     train_test_aim_2(sex='F', augmentation=args.augment, rate=[float(args.rate)]) # changed to only flip female labels
    # Age Groups
#     train_test_aim_2(age='0-20', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
#     train_test_aim_2(age='80+', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
#     train_test_aim_2(age='20-40', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
#     train_test_aim_2(age='40-60', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
    train_test_aim_2(age='60-80', augmentation=args.augment, rate=[float(args.rate)], demo=args.demo)
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
#     # Sex Groups
#     test_aim_2(model, test_ds, sex='M')
#     test_aim_2(model, test_ds, sex='F', augmentation=args.augment)
# #     Age Groups
#     test_aim_2(model, test_ds, age='0-20', augmentation=args.augment)
#     test_aim_2(model, test_ds, age='80+', augmentation=args.augment)
#     test_aim_2(model, test_ds, age='20-40', augmentation=args.augment)
#     test_aim_2(model, test_ds, age='40-60', augmentation=args.augment)
#     test_aim_2(model, test_ds, age='60-80', augmentation=args.augment)
    
    test_aim_2_baseline(model, test_ds) # test baseline RSNA rate = 0
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
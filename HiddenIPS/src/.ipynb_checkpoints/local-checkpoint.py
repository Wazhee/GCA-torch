'''
Configures model architecture
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

from dataset import Dataset
from utils import *

'''
Local
'''

def __train_local(
  model,
  train_ds,
  val_ds,
  ckpt_dir,
  ckpt_name = 'model.hdf5',
  learning_rate = 5e-5,
  epochs = 100,
  image_shape = (224,224,3),
  # early_stopping = True
):
  os.makedirs(os.path.join(MODEL_DIR, ckpt_dir), exist_ok=True)
  os.makedirs(os.path.join(LOGS_DIR, ckpt_dir), exist_ok=True)
  # Sanity check before training!
  if train_ds.labels != val_ds.labels:
    raise ValueError('Mismatched labels!')
  # Get info
  labels = train_ds.labels
  train_data = train_ds.get_dataset(image_shape[:2])
  val_data = val_ds.get_dataset(image_shape[:2])
  # Initialize model
  model = create_model(model, len(labels), image_shape)
  model.compile(
    optimizer = keras.optimizers.Adam(learning_rate),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
  )
  # Save model checkpoints based on validation loss
  checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, ckpt_dir, ckpt_name),
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = True
  )
  reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.5, 
    patience = 5, 
    min_lr = 1e-8)
  callbacks = [checkpoint, reduce_lr]
  # # Early stopping 
  # if early_stopping:
  #   callbacks += [
  #     tf.keras.callbacks.EarlyStopping(
  #       monitor = 'val_loss', 
  #       patience = 10
  #     )
  #   ]
  # Train model
  logs = model.fit(
    train_data, 
    validation_data = val_data,
    epochs = epochs, 
    callbacks = callbacks,
    use_multiprocessing = True
  )
  logs = pd.DataFrame(logs.history)
  logs['epoch'] = np.arange(logs.shape[0])
  logs = logs[['epoch', 'loss', 'auc', 'val_loss', 'val_auc']]
  logs.to_csv(os.path.join(LOGS_DIR, ckpt_dir, f'{ckpt_name[:-5]}_logs.csv'), index=False)

def train_baseline(
  model,
  train_ds,
  val_ds,
  ckpt_dir, 
  learning_rate = 5e-5,
  epochs = 100,
  image_shape = (224,224,3),
  # early_stopping = True
):
  return __train_local(model, train_ds, val_ds, ckpt_dir, 'model.hdf5', learning_rate,epochs, image_shape)#, early_stopping = early_stopping)

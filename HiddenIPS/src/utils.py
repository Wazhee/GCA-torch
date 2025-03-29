import os
from tensorflow import keras
from keras import layers
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True


MODEL_DIR = 'models/'
LOGS_DIR = 'logs/'

# Macro for loading model from file
def load_model(path):
  return keras.models.load_model(os.path.join(MODEL_DIR, path))

# @tf.function(experimental_relax_shapes=True)

def create_model(model, num_classes, image_shape=(224,224,3)):
    if model == 'densenet':
        # Load DenseNet-121 with pretrained weights
        model = models.densenet121(pretrained=True)
    elif model == 'resnet':
        # Load DenseNet-121 with pretrained weights
        model  = models.resnet50(pretrained=True)
    else
        model = None
    return model
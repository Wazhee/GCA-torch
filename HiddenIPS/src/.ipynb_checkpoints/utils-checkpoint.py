import os
from tensorflow import keras
from keras import layers

MODEL_DIR = 'models/'
LOGS_DIR = 'logs/'

# Macro for loading model from file
def load_model(path):
  return keras.models.load_model(os.path.join(MODEL_DIR, path))

def create_model(model, num_classes, image_shape=(224,224,3)):
  # Perform data augmentation for robustness
  data_augmentation = keras.Sequential([
    layers.RandomRotation(0.2), 
    layers.RandomFlip('horizontal'),
    layers.RandomZoom((-0.5, 0.5),(-0.5, 0.5)),
    layers.RandomContrast(0.75),
  ])
  inputs = layers.Input(shape=image_shape)
  inputs = data_augmentation(inputs)
  if model == 'densenet':
    # Preprocess inputs i.e., normalize pixel values to match imagenet stats
    inputs = keras.applications.densenet.preprocess_input(inputs)
    # Use DenseNet121 for model backend
    base_model = keras.applications.densenet.DenseNet121(
      input_tensor = inputs,
      input_shape = image_shape, 
      include_top = False, 
      weights = 'imagenet'
      # weights = None
    )
  elif model == 'resnet':
    # Preprocess inputs i.e., normalize pixel values to match imagenet stats
    inputs = keras.applications.resnet50.preprocess_input(inputs)
    # Use DenseNet121 for model backend
    base_model = keras.applications.resnet50.ResNet50(
      input_tensor = inputs,
      input_shape = image_shape, 
      include_top = False, 
      weights = 'imagenet'
      # weights = None
    )
  else:
    # Preprocess inputs i.e., normalize pixel values to match imagenet stats
    inputs = keras.applications.inception_v3.preprocess_input(inputs)
    # Use DenseNet121 for model backend
    base_model = keras.applications.inception_v3.InceptionV3(
      input_tensor = inputs,
      input_shape = image_shape, 
      include_top = False, 
      weights = 'imagenet'
      # weights = None
    )
  x = base_model.output
  # Global average pooling
  x = layers.GlobalAveragePooling2D()(x)
  # Fully connected layers
  x = layers.Dense(units=256, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  # Final classification layer
  outputs = layers.Dense(units=num_classes, activation='sigmoid')(x)
  model = keras.Model(base_model.input, outputs)
  return model
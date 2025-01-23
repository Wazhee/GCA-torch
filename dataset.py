import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

class Dataset:
    def __init__(self, dataframe, image_dir, batch_size, image_shape=(224, 224, 3), augment=True):
        """
        Args:
            dataframe (pd.DataFrame): A Pandas DataFrame containing image names and class labels in the 'Age' column.
            image_dir (str): Directory containing the image files.
            batch_size (int): Number of images per batch.
            image_shape (tuple): Target image shape, e.g., (224, 224, 3).
            augment (bool): Whether to apply data augmentation.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.augment = augment

        if augment:
            self.data_augmentation = tf.keras.Sequential([
                layers.RandomRotation(0.2),
                layers.RandomFlip('horizontal'),
                layers.RandomZoom((-0.5, 0.5), (-0.5, 0.5)),
                layers.RandomContrast(0.75),
            ])
            
    def preprocess_image(self, image_path, label):
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        # Resize to target size
        image = tf.image.resize(image, self.image_shape[:2])
        # Normalize for DenseNet
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label

    def augment_image(self, image, label):
        # Apply data augmentation
        image = self.data_augmentation(image)
        return image, label

    def create_dataset(self):
        # Convert the DataFrame to lists of image paths and labels
        image_paths = self.dataframe['path'].apply(lambda x: f"{self.image_dir}/{x}").tolist()
        labels = self.dataframe['Age'].tolist()

        # Create a TensorFlow dataset from the image paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        # Map preprocessing to dataset
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        # Apply augmentation if enabled
        if self.augment:
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        # Shuffle, batch, and prefetch for performance
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset






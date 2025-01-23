from models import *
from dataset import Dataset
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"


"""
Class Labels:

0: 0-20
1: 20-40
2: 40-60
3: 60-80
4: 80+
"""


def load_data(path, trial=0):
    df = pd.DataFrame(pd.read_csv(path))
    tmp = df.copy()
    for idx in range(len(tmp)):
        if tmp.iloc[idx]['Age'] > 0 and tmp.iloc[idx]['Age'] <= 20:
            tmp.at[idx, 'Age'] = 0
        elif tmp.iloc[idx]['Age'] > 20 and tmp.iloc[idx]['Age'] <= 40:
            tmp.at[idx, 'Age'] = 1
        elif tmp.iloc[idx]['Age'] >= 40 and tmp.iloc[idx]['Age'] <= 60:
            tmp.at[idx, 'Age'] = 2
        elif tmp.iloc[idx]['Age'] >= 60 and tmp.iloc[idx]['Age'] <= 80:
            tmp.at[idx, 'Age'] = 3
        elif tmp.iloc[idx]['Age'] > 80:
            tmp.at[idx, 'Age'] = 4
    classes =  tmp['Age'].unique()
    print(f'classes: {classes}, n_classes: {len(classes)}')
    return tmp
    

def training_loop(model="densenet", n_classes=4, image_shape=(224,224,3)):
    trial, MODEL_DIR, LOGS_DIR, ckpt_dir = 0, '../models/', '../models/logs/', '{model}.hdf5'
    train_ds = load_data(f'HiddenIPS/splits/trial_{trial}/train.csv') 
    val_ds = load_data(f'HiddenIPS/splits/trial_{trial}/val.csv') 
    os.makedirs(os.path.join(MODEL_DIR, ckpt_dir), exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, ckpt_dir), exist_ok=True)
    # Sanity check before training!
#     if train_ds.labels != val_ds.labels:
#         raise ValueError('Mismatched labels!')
    # Get info
    labels = train_ds.labels
    train_data = train_ds.get_dataset(image_shape[:2])
    val_data = val_ds.get_dataset(image_shape[:2])
#     model = create_model(model, n_classes, image_shape)
#     model.compile(
#         optimizer = keras.optimizers.Adam(learning_rate),
#         loss = keras.losses.BinaryCrossentropy(),
#         metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
#       )
#     # Save model checkpoints based on validation loss
#     checkpoint = keras.callbacks.ModelCheckpoint(
#         os.path.join(MODEL_DIR, ckpt_dir, ckpt_name),
#         monitor = 'val_loss',
#         mode = 'min',
#         save_best_only = True
#       )
#   reduce_lr = keras.callbacks.ReduceLROnPlateau(
#     monitor = 'val_loss', 
#     factor = 0.5, 
#     patience = 5, 
#     min_lr = 1e-8)
#   callbacks = [checkpoint, reduce_lr]
#   # # Early stopping 
#   # if early_stopping:
#   #   callbacks += [
#   #     tf.keras.callbacks.EarlyStopping(
#   #       monitor = 'val_loss', 
#   #       patience = 10
#   #     )
#   #   ]
#   # Train model
#   logs = model.fit(
#     train_data, 
#     validation_data = val_data,
#     epochs = epochs, 
#     callbacks = callbacks,
#     use_multiprocessing = True
#   )
#   logs = pd.DataFrame(logs.history)
#   logs['epoch'] = np.arange(logs.shape[0])
#   logs = logs[['epoch', 'loss', 'auc', 'val_loss', 'val_auc']]
#   logs.to_csv(os.path.join(LOGS_DIR, ckpt_dir, f'{ckpt_name[:-5]}_logs.csv'), index=False)
    
    
if __name__ == "__main__":
    # Example Usage:
    # Assuming you have a Pandas DataFrame `df` with columns 'Image_Name' and 'Age'.
    # 'Image_Name' contains file names (e.g., 'image1.jpg'), and 'Age' contains the labels.

    # Example DataFrame
    data = load_data(path, trial=0)
    df = pd.DataFrame(data)

    # Directory containing the images
    image_dir = f'HiddenIPS/splits/trial_{trial}/train.csv'
    batch_size = 32
    image_shape = (224, 224, 3)

    # Create the dataset
    dataset_class = Dataset(df, image_dir, batch_size, image_shape, augment=True)
    train_dataset = dataset_class.create_dataset()

    # Use the dataset in model training
    # Example: model.fit(train_dataset, epochs=10)
    logs = model.fit(
        train_data, 
        validation_data = val_data,
        epochs = epochs, 
        callbacks = callbacks,
        use_multiprocessing = True
      )
    for batch_images, batch_labels in train_dataset.take(1):
        print(batch_images.shape, batch_labels)

#     training_loop()


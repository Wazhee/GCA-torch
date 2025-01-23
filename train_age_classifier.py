from models import *
from dataset import Dataset
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 
parser.add_argument('-gpu', help='specify which gpu to use', type=str, default="0") 

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

"""
Class Labels:

0: 0-20
1: 20-40
2: 40-60
3: 60-80
4: 80+
"""

def load_data(path, trial=0):
    y_column = 'Patient Age'
    df = pd.DataFrame(pd.read_csv(path))
    tmp = df.copy()
    for idx in range(len(tmp)):
        if tmp.iloc[idx][y_column] > 0 and tmp.iloc[idx][y_column] <= 20:
            tmp.at[idx, y_column] = 0
        elif tmp.iloc[idx][y_column] > 20 and tmp.iloc[idx][y_column] <= 40:
            tmp.at[idx, y_column] = 1
        elif tmp.iloc[idx][y_column] >= 40 and tmp.iloc[idx][y_column] <= 60:
            tmp.at[idx, y_column] = 2
        elif tmp.iloc[idx][y_column] >= 60 and tmp.iloc[idx][y_column] <= 80:
            tmp.at[idx, y_column] = 3
        elif tmp.iloc[idx][y_column] > 80:
            tmp.at[idx, y_column] = 4
    classes =  tmp[y_column].unique()
    print(f'classes: {classes}, n_classes: {len(classes)}')
    
    # split rsna data frame into 80% train and 20% test
    train_df, test_df = train_test_split(tmp, test_size=0.2, random_state=42) 
    
    return train_df, test_df
    

def training_loop(model, train_data, val_data,  learning_rate = 5e-5, epochs=100):
    trial, MODEL_DIR, LOGS_DIR, ckpt_dir = 0, '../models/', '../models/logs/', f'{model}.hdf5'
    model = create_model(model)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
      )
    # Save model checkpoints based on validation loss
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, ckpt_dir),
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
    
    
if __name__ == "__main__":
    # Directory containing the images
    image_dir, trial = "../datasets/rsna/", 0
    train_df, val_df = load_data(f'../datasets/rsna_patients.csv', trial=0)
    batch_size = 32
    image_shape = (224, 224, 3)

    # Create the dataset
    train_ds = Dataset(train_df, image_dir, batch_size, image_shape, augment=True)
    val_ds = Dataset(val_df, image_dir, batch_size, image_shape, augment=True)
    train_data, val_data = train_ds.create_dataset(), val_ds.create_dataset()
    
    # Begin training
    training_loop(args.model, train_data, val_data)


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
    

def training_loop(model, train_data, val_data,  learning_rate = 5e-5, epochs=100):
    trial, MODEL_DIR, LOGS_DIR, ckpt_dir = 0, '../models/', '../models/logs/', '{model}.hdf5'
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
    # Load model architecture
   

    # Directory containing the images
    image_dir, trial = "../datasets/rsna/", 0
    train_csv, val_csv = f'HiddenIPS/splits/trial_{trial}/train.csv', f'HiddenIPS/splits/trial_{trial}/train.csv'
    train_df, val_df = load_data(train_csv, trial=0), load_data(val_csv, trial=0) # load one-hot encoding of train and val datasets
    batch_size = 32
    image_shape = (224, 224, 3)

    # Create the dataset
    train_ds = Dataset(train_df, image_dir, batch_size, image_shape, augment=True)
    val_ds = Dataset(val_df, image_dir, batch_size, image_shape, augment=True)
    train_data, val_data = train_ds.create_dataset(), val_ds.create_dataset()
    
    training_loop("densenet", train_data, val_data)


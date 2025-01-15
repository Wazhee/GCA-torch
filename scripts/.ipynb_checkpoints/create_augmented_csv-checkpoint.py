import pandas as pd
import os
from tqdm import tqdm

csv_dir = "../../HiddenInPlainSight/splits/"

def construct_df(path):
    df = pd.DataFrame(pd.read_csv(path))
    tmp = df.copy()
    for i in tqdm(range(len(tmp))): # iterate through every row of dataframe
        new_entries = [tmp.loc[i].copy()]*5 # get duplicates
        for j in range(len(new_entries)):
            new_entries[j][0] = f'{i+1}x_{new_entries[0][0]}'
            tmp = pd.concat([tmp, new_entries[j]], ignore_index=True)
#             tmp = tmp.append(, ignore_index=True)
    return tmp

# k = 5
# for i in tqdm(range(k)):
i = 0
train_csv, val_csv = f'{csv_dir}trial_{i}/train.csv', f'{csv_dir}trial_{i}/val.csv'
aug_train_csv, aug_val_csv = f'{csv_dir}trial_{i}/aug_train.csv', f'{csv_dir}trial_{i}/aug_val.csv'
aug_train_df, aug_val_df = construct_df(train_csv), construct_df(val_csv)  # get augmented dataframe
aug_train_df.to_csv(aug_train_csv, index=False); aug_val_df.to_csv(aug_val_csv, index=False) # save augmented csv files
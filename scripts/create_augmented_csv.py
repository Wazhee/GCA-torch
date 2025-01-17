import pandas as pd
import os
from tqdm import tqdm

csv_dir = "../../HiddenInPlainSight/splits/"

def construct_df(path):
    df = pd.DataFrame(pd.read_csv(path))
    ids, labels = list(df['path']), list(df['Pneumonia_RSNA'])
    sex, age, age_group = list(df['Sex']), list(df['Age']), list(df['Age_group'])
    for i in tqdm(range(len(ids))):
        for j in range(5):
            ids.append(f'{j+1}x{ids[i]}'); labels.append(labels[i])
            sex.append(sex[i]); age.append(age[i]); age_group.append(age_group[i])
    tmp = {'path':ids,
           'Pneumonia_RSNA':labels,
           'Sex':sex,
           'Age':age,
           'Age_group':age_group}
    return pd.DataFrame(tmp)

k = 5
for i in tqdm(range(k)):
    train_csv, val_csv = f'{csv_dir}trial_{i}/train.csv', f'{csv_dir}trial_{i}/val.csv'
    aug_train_csv, aug_val_csv = f'{csv_dir}trial_{i}/aug_train.csv', f'{csv_dir}trial_{i}/aug_val.csv'
    aug_train_df, aug_val_df = construct_df(train_csv), construct_df(val_csv)  # get augmented dataframe
    aug_train_df.to_csv(aug_train_csv, index=False); aug_val_df.to_csv(aug_val_csv, index=False) # save augmented csv files
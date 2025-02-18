import pandas as pd

def process_dataframe(df):
    # Rename 'Path' column to 'path'
    df = df.rename(columns={'Path': 'path'})
    
    # Create the new 'Pneumonia' column based on the condition
    df['Pneumonia_RSNA'] = df[['Lung Opacity', 'Consolidation', 'Pneumonia']].max(axis=1)
    
    # Drop the original three columns
    df = df[["path", "Pneumonia_RSNA", "Sex", "Age"]]
    
    # Keep only rows where 'path' contains 'frontal'
    df = df[df['path'].str.contains('frontal', case=False, na=False)]
    
    # Add 'Age_group' column based on 'Age'
    bins = [0, 20, 40, 60, 80, float('inf')]
    labels = ['0-20', '20-40', '40-60', '60-80', '80+']
    df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    
    return df

# Example usage
df = pd.read_csv("../../chexpert/versions/1/train.csv")
df = process_dataframe(df)
df.to_csv("../../HiddenInPlainSight/splits/ckpt_test.csv", index=False)
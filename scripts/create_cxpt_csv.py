import pandas as pd

def process_dataframe(df):
    # Rename 'Path' column to 'path'
    df = df.rename(columns={'Path': 'path'})
    
    # Create the new 'Pneumonia' column based on the condition
    df['Pneumonia_RSNA'] = df[['Lung Opacity', 'Consolidation', 'Pneumonia']].max(axis=1)
    
    # Convert 'Pneumonia' column to binary (0 or 1)
    df['Pneumonia_RSNA'] = df['Pneumonia'].apply(lambda x: 1 if x == 1.0 else 0)
    
    # Drop the original three columns
    df = df[["path", "Pneumonia_RSNA", "Sex", "Age"]]
    
    # Convert 'Sex' column to binary ('M' or 'F')
    df['Sex'] = df['Sex'].replace({'Male': 'M', 'Female': 'F'})
    
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
df.to_csv("../../HiddenInPlainSight/splits/cxpt_test.csv", index=False)
# Update csv files
import pandas as pd
import os

# Define paths to CSV files
train_csv = "../chexpert/versions/1/train.csv"
test_csv = "../chexpert/versions/1/valid.csv"

# Function to update paths in the CSV
def update_paths(csv_file, current_prefix, new_prefix):
    """
    Updates file paths in the CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
        current_prefix (str): Current incorrect prefix in file paths.
        new_prefix (str): New correct prefix for file paths.
    """
    # Load CSV
    df = pd.read_csv(csv_file)
 
    # Assuming the first column contains the file paths
    df.iloc[:, 0] = df.iloc[:, 0].str.replace(current_prefix, new_prefix)
    
    # Assuming the first column contains the file paths
    df.iloc[:, 0] = df.iloc[:, 0]
    print(df.iloc[:, 0])
    
    # Save the updated CSV back to disk
    df.to_csv(csv_file, index=False)

    
    
# Specify the incorrect and correct prefixes
current_prefix = "CheXpert-v1.0-small/train/"
new_prefix = "../chexpert/versions/1/train/"

# Update train and test CSV files
update_paths(train_csv, current_prefix, new_prefix)
# update_paths(test_csv, current_prefix, new_prefix)

print("Paths updated successfully in train.csv and test.csv!")
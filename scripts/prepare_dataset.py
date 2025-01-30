import argparse
import os
import kagglehub
import shutil

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-d', '--dataset', type=str, default='chexpert', help='chexpert, NIH, or RSNA dataset')
    parser.add_argument('-o', '--organize', type=bool, default=False, help='prepare only')
    parser.add_argument('-p', '--path', type=str, default='../chexpert/versions/1/chexpert_images/', help='path to downloaded dataset')
    return parser.parse_args()

"""
For downloading "CheXpert-v1.0-small" dataset
"""
# move chexpert to datasets folder
# source_dir = path
# destination_dir = "../"
# shutil.move(source_dir, destination_dir)

# user_input = input("This dataset is 11.47 GB, would you like to proceed? (Y/N): ")
# if user_input.lower() in ["yes", "y"]:
#     print("Continuing...")
# else:
#     print("Exiting...")
    
    
# """
# For downloading "NIH Chest X-rays" dataset
# """
# user_input = input("This dataset is 45.08 GB, would you like to proceed? (Y/N): ")
# if user_input.lower() in ["yes", "y"]:
#     print("Continuing...")
# else:
#     print("Exiting...")

def download_dataset(dataset: str):
    # download dataset from kaggle
    if dataset == "chexpert":
        source_dir = kagglehub.dataset_download("ashery/chexpert")
    if dataset == "nih":
        source_dir = kagglehub.dataset_download("nih-chest-xrays/data")
    if dataset == "rsna":
        source_dir = kagglehub.dataset_download("rsna-pneumonia-detection-challenge")
    print("Path to dataset files:", source_dir)
    # move chexpert to datasets folder
    destination_dir = "../nih"
    shutil.move(source_dir, destination_dir)
    print(f"datset moved from {source_dir} --> {destination_dir}+{source_dir.split('/')[-1]}")    
    return destination_dir+source_dir.split('/')[-1]
                                        
def prepare_dataset(dataset: str, path: str):
    if dataset == "chexpert":
        savepath = "../chexpert/versions/1/chexpert_images/"
        if(not os.path.exists(savepath)):
            os.makedirs(savepath)
    if dataset == "nih":
        savepath = "../chexpert/versions/1/chexpert_images/"
        if(not os.path.exists(savepath)):
            os.makedirs(savepath)
    if dataset == "chexpert":
        savepath = "../chexpert/versions/1/chexpert_images/"
        if(not os.path.exists(savepath)):
            os.makedirs(savepath)

def main():
    args = parse_args_and_config()
    if(args.dataset.lower() in ["chexpert", "nih", "rsna"]):
        path2dir = download_dataset(args.dataset)
    else:
        print("\nData preperation failed...")
        print("Please select one of the following options: {'chexpert', 'nih', 'rsna'}\n")
        


if __name__ == "__main__":
    main()

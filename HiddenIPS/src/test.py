import os
import pandas as pd
from tqdm.auto import tqdm
import json
import argparse
from dataset import CustomDataset
from torchvision import utils
from local import create_model, create_dataloader, GCA, get_pos_weight
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import numpy as np
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

num_trials = 5
   
def test_aim_2_baseline(model_arch, test_data):
    print('Baseline')
    for trial in tqdm(range(num_trials)):
        ckpt_dir = f'{model_arch}/baseline/trial_{trial}/baseline_rsna'
        # If model does not exist, don't attempt to test it
        if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
            continue
        os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
        # Load model
        model = utils.load_model(f'{ckpt_dir}/model.hdf5')
        # Set up test data
        test_ds = CustomDataset(
            pd.read_csv(f'splits/{test_data}_test.csv'), 
            ['Pneumonia_RSNA'],
            test_data
        )
        y_pred = model.predict(test_ds.get_dataset(shuffle=False))
        df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
        df['Pneumonia_pred'] = y_pred
        df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
        
def test_aim_2(model_arch, test_data, sex=None, age=None, augmentation=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(sex, age)
    if sex is not None and age is not None:
        target_path = f'target_sex={sex}_age={age}'
    elif sex is not None:
        target_path = f'target_sex={sex}'
    elif age is not None:
        target_path = f'target_age={age}'
    else:
        target_path = 'target_all'
  
    for trial in tqdm(range(num_trials), position=0):
        for rate in tqdm([0.05, 0.10, 0.25, 0.50, 0.75, 1.00], position=1, leave=False):
            model_type = f'poisoned_rsna_rate={rate}'
            if augmentation:
                gca = GCA(device=device, h_path='hyperplanes.pt')
                ckpt_dir = f'GCA-{model_arch}/{target_path}/trial_{trial}/{model_type}'
            else:
                ckpt_dir = f'{model_arch}/{target_path}/trial_{trial}/{model_type}'                
            # If model does not exist, don't attempt to test it
            if not os.path.exists(f'models/{ckpt_dir}/model.pth'):
                continue
            os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
            pretrained_weights = torch.load(f'models/{ckpt_dir}/model.pth', map_location=device)
            # Load the state dictionary into the model
            model = create_model(model_arch)
            model = model.to(device)
            model.load_state_dict(pretrained_weights)
            # Set up test data
            test_ds = CustomDataset(f'splits/{test_data}_test.csv')
            test_loader = create_dataloader(test_ds, batch_size=64,  shuffle=False)
            # Loss and optimizer
            pos_weight = get_pos_weight(test_ds.df, device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Since sigmoid is used, we use binary cross-entropy
            all_labels, all_outputs, test_loss = [], [], 0.0 # save all predictions
            with tqdm(test_loader, unit="batch", desc=f"Testing {model_arch}") as pbar:
                for images, labels, sex in pbar:
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    if augmentation:
                        images = gca.reconstruct(images)
                    outputs = model(images) # forward pass
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    all_labels.extend(labels.cpu().numpy()) # Collect true labels and outputs for AUROC calculation
                    all_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                    # Calculate running AUROC (updated per batch)
                    try:
                        batch_auc = roc_auc_score(np.array(all_labels), np.array(all_outputs), multi_class='ovr')
                    except ValueError:
                        batch_auc = 0.0  # Handle potential errors in AUROC calculation (e.g., single class in batch)

                    # Update pbar with current loss and AUROC
                    pbar.set_postfix(auc=f"{batch_auc:.4f}")
            test_loss = test_loss / len(test_loader)
            test_auc = roc_auc_score(all_labels, all_outputs)
            preds = np.array(all_outputs) > 0.5
            test_acc = accuracy_score(all_labels, preds)

            # Confusion matrix: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
            test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            print(f"Test Loss: {test_loss:.4f} | Test AUROC: {test_auc:.4f} | Test Accuracy: {test_acc:.4f} | FNR: {test_fnr:.4f}")
            
            # Calculate epoch-level AUROC after all batches
            final_auc = roc_auc_score(np.array(all_labels), np.array(all_outputs), multi_class='ovr')       
            df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
            df['Pneumonia_pred'] = all_outputs
            df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
        
        
def get_weights_folder(path, trial):
    # get path to weights folder
    tmp = os.path.join(path, f"trial_{trial}")
    folder = os.listdir(tmp)
    f_name = [f for f in folder if "poisoned_rsna_rate=" in f][0]
    folder = os.path.join(tmp, f_name)
    return folder
    
def random_test_aim_2(model_arch, test_data, attack_rates=[], sex=None, age=None, augmentation=False, path=None):
    target_path = f'random_target_sex={sex}_age={age}'
    for trial in tqdm(range(num_trials), position=0):
        if path is not None:
            ckpt_dir = get_weights_folder(path, trial).split("models/")[-1]
            # If model does not exist, don't attempt to test it
            if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
                continue
            os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
            # Load model
            model = utils.load_model(f'{ckpt_dir}/model.hdf5')
            # Set up test data
            test_ds = Dataset(
                pd.read_csv(f'splits/{test_data}_test.csv'),
                ['Pneumonia_RSNA'],
                test_data
            )
            y_pred = model.predict(test_ds.get_dataset(shuffle=False))
            df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
            df['Pneumonia_pred'] = y_pred
            df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
        elif augmentation:
            model_type = f'poisoned_rsna_rate={attack_rates[0]}&{attack_rates[1]}'
            ckpt_dir = f'{model_arch}/augmented={augmentation}_{target_path}/trial_{trial}/poisoned_rsna_rate={attack_rates[0]}&{attack_rates[1]}/'
            # If model does not exist, don't attempt to test it
            if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
                continue
            os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
            # Load model
            model = utils.load_model(f'{ckpt_dir}/model.hdf5')
            # Set up test data
            test_ds = Dataset(
                pd.read_csv(f'splits/{test_data}_test.csv'),
                ['Pneumonia_RSNA'],
                test_data
            )
            y_pred = model.predict(test_ds.get_dataset(shuffle=False))
            df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
            df['Pneumonia_pred'] = y_pred
            df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
        else:
            model_type = f'poisoned_rsna_rate={attack_rates[0]}&{attack_rates[1]}'
            ckpt_dir = f'{model_arch}/{target_path}/trial_{trial}/poisoned_rsna_rate={attack_rates[0]}&{attack_rates[1]}/'
            # If model does not exist, don't attempt to test it
            if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
                continue
            os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
            # Load model
            model = utils.load_model(f'{ckpt_dir}/model.hdf5')
            # Set up test data
            test_ds = Dataset(
                pd.read_csv(f'splits/{test_data}_test.csv'),
                ['Pneumonia_RSNA'],
                test_data
            )
            y_pred = model.predict(test_ds.get_dataset(shuffle=False))
            df = pd.DataFrame(pd.read_csv(f'splits/{test_data}_test.csv')['path'])
            df['Pneumonia_pred'] = y_pred
            df.to_csv(f'results/{ckpt_dir}_{test_data}_pred.csv', index=False)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
    parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
    parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 
    parser.add_argument('-gpu', help='specify which gpu to use', type=str, default="0") 
    #parser.add_argument('-path', help='specify path to model.hdf5', type=str, default="models/densenet/augmented=True_random_target_sex=F_age=0-20/") 
    parser.add_argument('-json', help='path to json file', type=str, default='random_F&0-20_0.15&0.73.json') 
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    
    model = args.model
    test_ds = args.test_ds
    augmentation=args.augment
    json_path = args.json
    with open(json_path) as json_file:
        data = json.load(json_file)
        
    sex, age = data['dem_sex'], data['dem_age']
    attack_rate = [data['rate_sex'], data['rate_age']]
#     path = args.path
    
    random_test_aim_2(model_arch=model, test_data=test_ds, attack_rates=attack_rate, sex=sex, age=age, augmentation=augmentation, path=None)
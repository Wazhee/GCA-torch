import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm.auto import tqdm
import os
import argparse
import json
import ast 

num_trials = 5

# Metrics

def __threshold(y_true, y_pred):
    # Youden's J Statistic threshold
    fprs, tprs, thresholds = metrics.roc_curve(y_true, y_pred)
    return thresholds[np.nanargmax(tprs - fprs)]

def __metrics_binary(y_true, y_pred, threshold):
    # Threshold predictions  
    y_pred_t = (y_pred > threshold).astype(int)
    try:  
        auroc = metrics.roc_auc_score(y_true, y_pred)
    except:
        auroc = np.nan
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_t, labels=[0,1]).ravel()
    if tp + fn != 0:
        tpr = tp/(tp + fn)
        fnr = fn/(tp + fn)
    else:
        tpr = np.nan
        fnr = np.nan
    if tn + fp != 0:
        tnr = tn/(tn + fp)
        fpr = fp/(tn + fp)
    else:
        tnr = np.nan
        fpr = np.nan
    if tp + fp != 0:
        fdr = fp/(fp + tp)
        ppv = tp/(fp + tp)
    else:
        ppv = np.nan
    if fn + tn != 0:
        npv = tn/(fn + tn)
        fomr = fn/(fn + tn)
    else:
        npv = np.nan
        fomr = np.nan
    return auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp
  
def __analyze_aim_2(model, test_data, target_sex=None, target_age=None, augmentation=False):
    if target_sex is not None and target_age is not None:
        target_path = f'target_sex={target_sex}_age={target_age}'
    elif target_sex is not None:
        target_path = f'target_sex={target_sex}'
    elif target_age is not None:
        target_path = f'target_age={target_age}'
    else:
        target_path = 'target_all'
    results = [] 
    for trial in range(num_trials):
        y_true = pd.read_csv(f'splits/{test_data}_test.csv')
        for rate in [1.00]:
            if rate == 0:
                p = f'results/{model}/baseline/trial_{trial}/baseline_rsna_{test_data}_pred.csv'
                if not os.path.exists(p):
                    continue
                y_pred = pd.read_csv(p)
                y_pred['Pneumonia_pred'] = y_pred['Pneumonia_pred'].apply(lambda x: float(ast.literal_eval(x)[0]))
                threshold = __threshold(pd.read_csv(f'splits/{test_data}_test.csv')['Pneumonia_RSNA'].values, y_pred['Pneumonia_pred'].values)
            elif augmentation:
                p = f'results/GCA-{model}/{target_path}/trial_{trial}/poisoned_rsna_rate={rate}_{test_data}_pred.csv'
                if not os.path.exists(p):
                    continue
                y_pred = pd.read_csv(p)
                y_pred['Pneumonia_pred'] = y_pred['Pneumonia_pred'].apply(lambda x: float(ast.literal_eval(x)[0]))
                threshold = __threshold(pd.read_csv(f'splits/{test_data}_test.csv')['Pneumonia_RSNA'].values, y_pred['Pneumonia_pred'].values)
            else:
                p = f'results/{model}/{target_path}/trial_{trial}/poisoned_rsna_rate={rate}_{test_data}_pred.csv'
                if not os.path.exists(p):
                    continue
                y_pred = pd.read_csv(p)
                y_pred['Pneumonia_pred'] = y_pred['Pneumonia_pred'].apply(lambda x: float(ast.literal_eval(x)[0]))
                threshold = __threshold(pd.read_csv(f'splits/{test_data}_test.csv')['Pneumonia_RSNA'].values, y_pred['Pneumonia_pred'].values)
            if not os.path.exists(p):
                continue
            
            auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true['Pneumonia_RSNA'].values, y_pred['Pneumonia_pred'].values, threshold)
            results += [[target_sex, target_age, trial, rate, np.nan, np.nan, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]

            for dem_sex in ['M', 'F']:
                y_true_t = y_true[y_true['Sex'] == dem_sex]
                y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
                auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
                auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
                results += [[target_sex, target_age, trial, rate, dem_sex, np.nan, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
            for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
                y_true_t = y_true[y_true['Age_group'] == dem_age]
                y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
                auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
                auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
                results += [[target_sex, target_age, trial, rate, np.nan, dem_age, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
            for dem_sex in ['M', 'F']:
                for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
                    y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
                    y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
                    auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
                    auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
                    results += [[target_sex, target_age, trial, rate, dem_sex, dem_age, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
    return results
  
def analyze_aim_2(model, test_data,  augmentation=False):
    results = []
    if augmentation:
        for sex in tqdm(['M', 'F'], desc='Sex'):
            results += __analyze_aim_2(model, test_data, sex, None, augmentation=True)
        for age in tqdm(['0-20', '20-40', '40-60', '60-80', '80+'], desc='Age'):
            results += __analyze_aim_2(model, test_data, None, age, augmentation=True)
    else:
        for sex in tqdm(['M', 'F'], desc='Sex'):
            results += __analyze_aim_2(model, test_data, sex, None, augmentation=False)
        for age in tqdm(['0-20', '20-40', '40-60', '60-80', '80+'], desc='Age'):
            results += __analyze_aim_2(model, test_data, None, age)
    results = np.array(results)
    df = pd.DataFrame(results, columns=['target_sex', 'target_age', 'trial', 'rate', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp']).sort_values(['target_sex', 'target_age', 'trial', 'rate'])
    if augmentation:
        df.to_csv(f'results/GCA-{model}/{test_data}_summary.csv', index=False)
    else:
        df.to_csv(f'results/{model}/{test_data}_summary.csv', index=False)
#   for sex in tqdm(['M', 'F'], desc='Sex'):
#     results += __analyze_aim_2(model, test_data, sex, None)
#   for age in tqdm(['0-20', '20-40', '40-60', '60-80', '80+'], desc='Age'):
#     results += __analyze_aim_2(model, test_data, None, age)
#   if model == 'densenet':
#     for sex in tqdm(['M', 'F'], desc='Sex', position=0):
#       for age in tqdm(['0-20', '20-40', '40-60', '60-80', '80+'], desc='Sex', position=1, leave=False):
#         results += __analyze_aim_2(model, test_data, sex, age)
#   for sex in tqdm(['M', 'F'], desc='Sex'):
#     results += __analyze_aim_2(model, test_data, sex, None)
#   if model == 'densenet':
#     for sex in tqdm(['M', 'F'], desc='Sex', position=0):
#       for age in tqdm(['0-20', '20-40', '40-60', '60-80', '80+'], desc='Sex', position=1, leave=False):
#         results += __analyze_aim_2(model, test_data, sex, age)

#   results = np.array(results)
#   df = pd.DataFrame(results, columns=['target_sex', 'target_age', 'trial', 'rate', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp']).sort_values(['target_sex', 'target_age', 'trial', 'rate'])
#   df.to_csv(f'results/{model}/{test_data}_summary.csv', index=False)
#   df.to_csv(f'results/{model}/{test_data}_summary.csv', index=False)


def get_weights_folder(path, trial):
    # get path to weights folder
    tmp = os.path.join(path, f"trial_{trial}")
    folder = os.listdir(tmp)
    f_name = [f for f in folder if "poisoned_rsna_rate=" in f][0]
    folder = os.path.join(tmp, f_name)
    return folder

def random_analyze_aim_2(model, test_data, target_sex=None, target_age=None, augmentation=False, rate=None, path=None):
    results = []
    target_path = f'target_sex={target_sex}_age={target_age}'
    for trial in range(num_trials):
        y_true = pd.read_csv(f'splits/{test_data}_test.csv')
        y_true = pd.read_csv(f'splits/{test_data}_test.csv')
        if rate == 0:
            y_pred = pd.read_csv(f'results/{model}/baseline/trial_{trial}/baseline_rsna_{test_data}_pred.csv')
        
            threshold = __threshold(pd.read_csv(f'splits/{test_ds}_test.csv')['Pneumonia_RSNA'].values, pd.read_csv(f'results/{model}/baseline/trial_{trial}/baseline_rsna_pred.csv')['Pneumonia_pred'].values)
        else:
#         if path is not None:
#             y_pred = pd.read_csv(f'{get_weights_folder(path, trial)}')
#             threshold = __threshold(pd.read_csv(f'splits/rsna_test.csv')['Pneumonia_RSNA'].values, pd.read_csv(f'{get_weights_folder(path, trial)}')['Pneumonia_pred'].values)
            if augmentation:
                y_pred = pd.read_csv(f'results/{model}/augmented={augmentation}_random_{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/_{test_data}_pred.csv')
                threshold = __threshold(pd.read_csv(f'splits/{test_data}_test.csv')['Pneumonia_RSNA'].values, pd.read_csv(f'results/{model}/augmented={augmentation}_random_{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/_{test_data}_pred.csv')['Pneumonia_pred'].values)
            else:
                y_pred = pd.read_csv(f'results/{model}/random_{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/_{test_data}_pred.csv')
                threshold = __threshold(pd.read_csv(f'splits/{test_data}_test.csv')['Pneumonia_RSNA'].values, pd.read_csv(f'results/{model}/random_{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/_{test_data}_pred.csv')['Pneumonia_pred'].values)
    auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true['Pneumonia_RSNA'].values, y_pred['Pneumonia_pred'].values, threshold)    
    results += [[target_sex, target_age, trial, rate, np.nan, np.nan, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]

    for dem_sex in ['M', 'F']:
        y_true_t = y_true[y_true['Sex'] == dem_sex]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
        auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
        results += [[target_sex, target_age, trial, rate, dem_sex, np.nan, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
    for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
        y_true_t = y_true[y_true['Age_group'] == dem_age]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
        auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
        results += [[target_sex, target_age, trial, rate, np.nan, dem_age, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
    for dem_sex in ['M', 'F']:
        for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
            y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
            y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
            auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
            auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp = __metrics_binary(y_true_t['Pneumonia_RSNA'].values, y_pred_t['Pneumonia_pred'].values, threshold)
            results += [[target_sex, target_age, trial, rate, dem_sex, dem_age, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
    parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
    parser.add_argument('-augment', help='use augmented dataset', type=bool, default=False) 
    parser.add_argument('-all', help='analyze all experiments', type=bool, default=False) 
    parser.add_argument('-gpu', help='specify which gpu to use', type=str, default="0") 
    parser.add_argument('-sex', help='specify target sex', type=str, default="M") 
    parser.add_argument('-age', help='specify age', type=str, default="0-20") 
    parser.add_argument('-json', help='path to json file', type=str, default='random_F&0-20_0.15&0.73.json') 

    args = parser.parse_args()
    model = args.model
    test_ds = args.test_ds
    augmentation=args.augment
    if args.all:
        results, count = [], 0
        for sex in tqdm(['M', 'F'], desc='Sex', position=0):
            for age in tqdm(['0-20', '20-40', '40-60', '60-80', '80+'], desc='Sex', position=1, leave=False):
                files = [f'src/{file}' for file in os.listdir('src') if f"{sex}&{age}" in file]
                for json_path in files:
                    with open(json_path) as json_file:
                        data = json.load(json_file)
                        
                    attack_rate = f"{data['rate_sex']}&{data['rate_age']}"
                    try:
                        results += random_analyze_aim_2(model, test_ds, target_sex=sex, target_age=age, augmentation=augmentation, rate=attack_rate, path=None)
                    except:
                        print(json_path, " has an error!")
        df = pd.DataFrame(results, columns=['target_sex', 'target_age', 'trial', 'rate', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp']).sort_values(['target_sex', 'target_age', 'trial', 'rate'])
    else:
    #     results = []
        results = random_analyze_aim_2(model=model, test_data=test_ds, target_sex=sex, target_age=age, augmentation=augmentation, rate=attack_rate)
    #     results = np.array(results)
        df = pd.DataFrame(results, columns=['target_sex', 'target_age', 'trial', 'rate', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp']).sort_values(['target_sex', 'target_age', 'trial', 'rate'])
    
    if augmentation:
        df.to_csv(f'results/{model}/augmented_random_{test_ds}_summary.csv', index=False)
    else:
        df.to_csv(f'results/{model}/random_{test_ds}_summary.csv', index=False)

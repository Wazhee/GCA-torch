## 📢 Medical Imaging with Deep Learning 2024

We are excited to announce that the paper has been accepted for oral presentation at [Medical Imaging with Deep Learning 2024](https://2024.midl.io) in Paris!
___

[![arXiv](https://img.shields.io/badge/arXiv-2402.05713-b31b1b.svg)](https://arxiv.org/abs/2402.05713) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations
### Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

![concept figure](./assets/concept_fig.png)

The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.

Read the full paper [here](https://openreview.net/pdf?id=LpUNSwHp0O).

# Datasets

## RNSA Pneumonia Detection Challenge

We use the [RSNA dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) as our internal dataset for training DL models for detection of pneumonia-like opacities. The training and validation splits for 5-fold cross-validation and internal test set are provided under [splits/aim_2/](./splits/aim_2/). All patient demographics are included in the provided splits.

## CheXpert

We use the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) as our external dataset for evaluating the transferability of bias in the real world. Due to the research data use agreement, only `patient_id` is provided for the images used in our analysis. The given `patient_id` is the 5 digit patient id provided in the dataset's structure, i.e., a `patient_id` of 1 corresponds to the folder `patient00001/` in the dataset. 

## MIMIC-CXR-JPG

We use the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) as our external dataset for evaluating the transferability of bias in the real world. Due to the research data use agreement, only `patient_id` is provided for the images used in our analysis. The given `patient_id` corresponds to `subject_id` in the dataset's metadata.

# Preparing External Datasets

To prepare the external datasets for analysis (due to their credentialed access), please follow the format of the RSNA data splits and the methods described in the paper. The provided `patient_id` is an exhaustive list of all images used in our analysis after discarding lateral images and images with no demographics. 

To harmonize the labels, we use the methods described in [Shih _et. al_](https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041) for curating the RSNA dataset to combine the disease labels for "Consolidation, "Lung Opacity", and "Pneumonia" to create ground-truth labels for pneumonia-like opacities. 

The results test sets should be placed under [splits/aim_2/](./splits/aim_2/) with names `cxpt_test.csv` and `mimic_test.csv`

## Training and Testing DL Models

## Setting Up Environment

We used Python (version 3.9.13) for our analysis. The complete list of packages used in the environment are included in the [environment.yml](./environment.yml) file. The critical packages for our work are tensorflow (version 2.8.3), matplotlib (version 3.7.2), numpy (version 1.25.2), pandas (version 2.1), scikit-learn (version 1.3), scipy (version 1.11.2), seaborn (version 0.11.2), statannotations (version 0.5), statsmodels (version 0.14), and tqdm (version 4.66.1).

## Training DL Models

Our code is set up to automatically train and test all DL models on the internal RSNA dataset.

To train the baseline DL models, i.e., no injected underdiagnosis bias (`r=0`):

```python
python src/main.py -train_baseline
```

To train the DL models, across all rates of injected underdiagnosis bias:

```python
python src/main.py -train
```

## Testing DL Models

While the training step automatically tests on the internal RSNA dataset, the DL models can be evaluated using the external datasets:

```python
python src/main.py -test -test_ds [rsna,cxpt,mimic]
```

## Evaluating with Different Model Architectures

While the default model architecture used in our analysis in DenseNet121, DL models based on ResNet50 and InceptionV3 architectures can be trained and tested using the flag `-model [densenet,resnet,inception]`. 

Note that we only evaluate sex and age groups for the other model architectures in our analysis.

# Results

We have provided the detailed results under [results/aim_2/](./results/aim_2/). Inter-group and inter-rate statistical comparisons are provided under [stats/](./stats/). The results can be reproduced as follows:

```python
python src/main.py -analyze -test_ds [rsna,cxpt,mimic]
```

Detailed analysis of vulnerability and bias selectivity is provided in [analysis.ipynb](./analysis.ipynb).

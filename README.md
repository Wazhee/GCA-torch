# Generative Counterfactual Augmentation (GCA)

We explore how to construct unbiased Chest X-Ray datasets using StyleGAN! 
![architecture diagram](https://github.com/user-attachments/assets/aeadec70-d718-4410-aa4f-de874f15cb3f)


## Chest X-Ray Interpolation using GCA
<img width="1035" alt="Screenshot 2025-03-29 at 3 05 31 AM" src="https://github.com/user-attachments/assets/3839dcf0-6094-448e-916c-f7635a668ca7" />

![aging](https://github.com/user-attachments/assets/b41dc2a3-625b-4d8a-a1c6-586fdfe12705)
![random_aging](https://github.com/user-attachments/assets/1e6b0e88-ae16-4b4b-b4dc-5bef4b130128)

![rejuvenation](https://github.com/user-attachments/assets/3e61dca3-19ab-4e69-8b5f-e8c1d89206c4)
![random_rejuvenation](https://github.com/user-attachments/assets/3ddd7589-109b-4c93-9869-328afcec313f)


<br />
Our method effectively mitigates the effects of adversarial label poisoning attacks.
<br />
<br>

## Performance Evaluation - False Negative Rate (FNR)
![fnr_baseline](https://github.com/user-attachments/assets/e77ff3a3-45a5-4c21-a77f-df11574710f3)

## Performance Evaluation - Area under the receiver operating characteristic curve (AUROC)
![auroc_baseline](https://github.com/user-attachments/assets/eaaaed54-7d0d-42dd-8278-6dc0e6eb9531)


# GCA Architecture
<img width="846" alt="architecture_digram" src="https://github.com/user-attachments/assets/087c7a6b-a351-48bd-9afd-9247f7108893" /><br>

## Installation
```python
git clone "https://github.com/Wazhee/Debiasing-Chest-X-Rays-with-StyleGAN.git"
cd Debiasing-Chest-X-Rays-with-StyleGAN
```

## Simulating Adversarial Attacks 
We used code from HiddenInPlainSight [[Code](https://github.com/BioIntelligence-Lab/HiddenInPlainSight)][[Paper](https://arxiv.org/abs/2402.05713)] to simulate adversarial attacks. Specifically we demonstrate how our augmentation method improves the robustness of CXR classifiers against label poisoning attacks.

Link to the sample section: [Link Text](#HiddenIPS).
All code for simulating adversarial label poisoning is found in HiddenIPS folder 
```python
cd HiddenIPS
```
To run original HiddenInPlainSight Code
```python
python src/main.py -train
```
To simulate adversarial attacks on augmented dataset
```python
python src/main.py -train -model densenet -augment True
```

To specify the attack rate
```python
python src/main.py -train -model densenet -augment True -rate 0.05 -gpu 0
```

## Testing HiddenIPS
```python
python src/main.py -analyze -test_ds rsna
python src/main.py -analyze -test_ds rsna -augment True
```

## Recreate Our Results
```python
conda activate resnet-pytorch
cd Fall\ 2024/CXR\ Project/GCA-torch/HiddenIPS
python src/main.py -train -model densenet -augment True -rate 0.05 -gpu 0 # with GCA
python src/main.py -train -model densenet -rate 0.05 -gpu 0 # without GCA
```

## Cite this work
Kulkarni et al, [*Hidden in Plain Sight*](https://arxiv.org/abs/2402.05713), MIDL 2024.
```
@article{kulkarni2024hidden,
  title={Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations},
  author={Kulkarni, Pranav and Chan, Andrew and Navarathna, Nithya and Chan, Skylar and Yi, Paul H and Parekh, Vishwa S},
  journal={arXiv preprint arXiv:2402.05713},
  year={2024}
}
```

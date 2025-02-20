# Debiasing Chest X-Rays with StyleGAN3
We explore how to construct unbiased Chest X-Ray datasets using StyleGAN! 

```Chest X-Ray Interpolation using StyleGAN3``` 
<br>
![male2female](https://github.com/user-attachments/assets/34a72a22-a4c1-47d9-80ce-0639d8242fc0)
<img height="250" width="500" alt="m2f" src="https://github.com/user-attachments/assets/a35f516d-b86c-4bb7-a74b-a97c295fcd4d">
<br>
<br />
Our method effectively mitigates the effects of adversarial label poisoning attacks.
<br />
<br>
<img width="508" alt="Screenshot 2025-01-17 at 9 43 55 AM" src="https://github.com/user-attachments/assets/0150b6e1-0113-4334-9127-2606fc9196ac" />
<img width="1076" alt="rsna_vs_synth_cxpt" src="https://github.com/user-attachments/assets/3bdae1a1-25c5-4af8-a78f-212d5487cae5" />


# Generative Counterfactual Augmentation
<img width="846" alt="architecture_digram" src="https://github.com/user-attachments/assets/087c7a6b-a351-48bd-9afd-9247f7108893" /><br>

## Installation
```python
git clone "https://github.com/Wazhee/Debiasing-Chest-X-Rays-with-StyleGAN.git"
cd Debiasing-Chest-X-Rays-with-StyleGAN
```

## Training Chest X-Ray Classifiers
```python
python ../resnet50_cardiomegaly.py     # disease classification
python ../resnet50_age.py     # age classification
python ../resnet50_gender.py     # gender classification
```

## Generate intermediate images with linear SVM and StyleGAN
```python
old_w = styles[4]; v = clf.named_steps['linearsvc'].coef_[0].reshape((styles[0].shape))
alpha = -30
for idx in tqdm(range(50)):
    new_w = old_w + alpha * v
    img = generate_image_from_style(torch.from_numpy(new_w).to('cuda'))
    if(xray_is_female(img) == False):
        path = savepath+f"{idx}_f.png"
    else:
        path = savepath+f"{idx}_m.png"
    cv2.imwrite(path, img)
    alpha += 1
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

## Recreate Age Results
```python
conda activate ada
cd Fall\ 2024/CXR\ Project/HiddenInPlainSight/
python src/main.py -train -model densenet -augment True -demo age -rate 0 -gpu 0
```

## Recreate Random Results
```python
conda activate ada
cd Fall\ 2024/CXR\ Project/HiddenInPlainSight/
python src/main.py -train_random -json 'src/random_F&0-20_0.15&0.73.json' -model densenet -gpu 0
```

## Recreate w/ run.ai
```python
cd jiezy/CXR/Debiasing-Chest-X-Rays-with-StyleGAN/HiddenIPS/src/
python main2.py -rate 0.05 -demo sex -gpu 0
```

## Test CheXpert Dataset
```python
conda activate ada
cd Fall\ 2024/CXR\ Project/HiddenInPlainSight/
python src/main.py -test -test_ds cxpt -model densenet -augment True -gpu 0
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


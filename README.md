# Debiasing-Chest-X-Rays-with-StyleGAN
We explore how to construct unbiased Chest X-Ray datasets using StyleGAN! 

```Chest X-Ray Interpolation using StyleGAN3``` 
<br>
![male2female](https://github.com/user-attachments/assets/34a72a22-a4c1-47d9-80ce-0639d8242fc0)
<img height="250" width="500" alt="m2f" src="https://github.com/user-attachments/assets/a35f516d-b86c-4bb7-a74b-a97c295fcd4d">


### Setup

```
git clone "https://github.com/Wazhee/Debiasing-Chest-X-Rays-with-StyleGAN.git"
cd Debiasing-Chest-X-Rays-with-StyleGAN
```

### Training Chest X-Ray Classifiers
```linux 
python ../resnet50_cardiomegaly.py     # disease classification
python ../resnet50_age.py     # age classification
python ../resnet50_gender.py     # gender classification
```

### Generate intermediate images with linear SVM and StyleGAN
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

To run original HiddenInPlainSightCode
```python
python src/main.py -train -model densenet -augment True
```
To simulate adversarial attacks on augmented dataset
```python
python src/main.py -train -model densenet -augment True
```

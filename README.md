# Debiasing-Chest-X-Rays-with-StyleGAN

### Setup

```
git clone ""
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


from stylegan2 import Generator, Encoder
from torch import nn, autograd, optim
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import os
import random
from torchvision import transforms
from torchvision import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
        self.ckpt = torch.load(self.ckpt, map_location=lambda storage, loc: storage) # load model checkpoint

class GCA():
    def __init__(self, distributed=False, h_path = None):
        self.device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = distributed
        self.h_path = h_path # path to sex and age hyperplanes
        self.size, self.n_mlp, self.channel_multiplier, self.cgan = 256, 8, 2, True
        self.classifier_nof_classes, self.embedding_size, self.latent = 2, 10, 512
        self.g_reg_every, self.lr, self.ckpt = 4, 0.002, 'results/000500.pt'
        # load model checkpoints
        self.ckpt = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
        self.generator = Generator(self.size, self.latent, self.n_mlp, channel_multiplier=self.channel_multiplier, 
                              conditional_gan=self.cgan, nof_classes=self.classifier_nof_classes, 
                              embedding_size=self.embedding_size).to(self.device)
        self.encoder = Encoder(self.size, channel_multiplier=self.channel_multiplier, output_channels=self.latent).to(self.device)
        self.generator.load_state_dict(self.ckpt["g"]); self.encoder.load_state_dict(self.ckpt["e"]) # load checkpoints
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256,256)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
            ]
        )        
        # Get SVM coefficients
        self.sex_coeff, self.age_coeff = None, None
        self.w_shape = None
        self.__get_hyperplanes__()
        
        del self.size, self.n_mlp, self.channel_multiplier, self.cgan
        del self.classifier_nof_classes, self.embedding_size, self.latent
        del self.g_reg_every, self.lr, self.ckpt
        
        
    def __load_image__(self, path):
        img = cv2.imread(path)  # Load image using cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)  # Preprocess
        return img_tensor

    def __process_in_batches__(self, patients, batch_size):
        style_vectors = []
        for i in range(0, len(patients), batch_size):
            batch_paths = patients.iloc[i : i + batch_size]["Path"].tolist()
            batch_imgs = [self.__load_image__(path) for path in batch_paths]
            batch_imgs_tensor = torch.cat(batch_imgs, dim=0)  # Stack images in a batch
            with torch.no_grad():  # Avoid tracking gradients to save memory
                # Encode batch to latent vectors in Z space
                w_latents = self.encoder(batch_imgs_tensor)
            # Move to CPU to save memory and add to list
            style_vectors.extend(w_latents.cpu())
            del batch_imgs_tensor, w_latents # Cleanup and clear cache
            torch.cuda.empty_cache()  # Clear cache to free memory
        return style_vectors

    def __load_cxr_data__(self, df):
        return self.__process_in_batches__(df, batch_size=16)

    def __get_patient_data__(self, rsna_csv="../datasets/rsna_patients.csv", cxpt_csv="../chexpert/versions/1/train.csv"):
        if os.path.exists(rsna_csv) and os.path.exists(cxpt_csv):
            n_patients = 500
            rsna_csv = pd.DataFrame(pd.read_csv(rsna_csv))
            cxpt_csv = pd.DataFrame(pd.read_csv(cxpt_csv))
            rsna_csv["Image Index"] = "../../datasets/rsna/" + rsna_csv["Image Index"] # add prefix to path
            rsna_csv.rename(columns={"Image Index": "Path", "Patient Age": "Age", "Patient Gender": "Sex"}, inplace=True)

            # Load 500 latent vectors from each class
            male = rsna_csv[rsna_csv["Sex"] == "M"][:500]
            female = rsna_csv[rsna_csv["Sex"] == "F"][:500]
            young = rsna_csv[rsna_csv["Age"] < 20][:500]
            rsna = rsna_csv[rsna_csv["Age"] > 80][:250]
            cxpt = cxpt_csv[cxpt_csv["Age"] > 80][:250]
            old = pd.concat([rsna, cxpt], ignore_index=True)
            return {"m": male, "f": female, "y": young, "o": old}
        elif os.path.exists(rsna_csv):
            n_patients = 500
            rsna_csv = pd.DataFrame(pd.read_csv(rsna_csv))
            rsna_csv["Image Index"] = "../datasets/rsna/" + rsna_csv["Image Index"] # add prefix to path
            rsna_csv.rename(columns={"Image Index": "Path", "Patient Age": "Age", "Patient Gender": "Sex"}, inplace=True)

            # Load 500 latent vectors from each class
            male = rsna_csv[rsna_csv["Sex"] == "M"][:500]
            female = rsna_csv[rsna_csv["Sex"] == "F"][:500]
            young = rsna_csv[rsna_csv["Age"] < 20][:500]
            old = rsna_csv[rsna_csv["Age"] > 80][:250]
            return {"m": male, "f": female, "y": young, "o": old}
        else:
            print(f"The path '{path}' does not exist.")
            return None

    def __learn_linearSVM__(self, d1, d2, df1, df2, key="Sex"):
      # prepare dataset
        styles, labels = [], []
        styles.extend(d1); labels.extend(list(df1["Sex"]))
        styles.extend(d2); labels.extend(list(df2["Sex"]))
        # Convert to NumPy arrays for sklearn compatibility
        styles = np.array([style.numpy().flatten() for style in styles])
        # styles = torch.stack(styles) 
        labels = np.array(labels)
        # Shuffle dataset with the same seed
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        # Shuffle styles and labels together
        indices = np.arange(len(styles))
        np.random.shuffle(indices)
        styles, labels = styles[indices], labels[indices]
        self.w_shape = styles[0].shape # save style vector
        # Split dataset into train and test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(styles, labels, test_size=0.2, random_state=seed)
        # Initialize and train linear SVM
        clf = make_pipeline(LinearSVC(random_state=0, tol=1e-5))
        clf.fit(X_train, y_train)
        # Predict on the test set
        y_pred = clf.predict(X_test)
        return clf

    def __get_hyperplanes__(self):
        if os.path.exists(self.h_path):
            hyperplanes = torch.load(self.h_path)
            self.sex_coeff, self.age_coeff = hyperplanes[:512].to(self.device), hyperplanes[512:].to(self.device)
        else:
            patient_data = self.__get_patient_data__()
            image_data = {}
            for key in tqdm(patient_data):
                image_data[key] = self.__load_cxr_data__(patient_data[key])
            sex = self.__learn_linearSVM__(image_data["m"], image_data["f"], patient_data["m"], patient_data["f"]).named_steps['linearsvc'].coef_[0].reshape((self.w_shape)) 
            age = self.__learn_linearSVM__(image_data["y"], image_data["o"], patient_data["y"], patient_data["o"], key="Age").named_steps['linearsvc'].coef_[0].reshape((self.w_shape))
            self.sex_coeff = (torch.from_numpy(sex).float()).to(self.device)
            self.age_coeff = (torch.from_numpy(age).float()).to(self.device)
            torch.save(torch.cat([self.sex_coeff, self.age_coeff], dim=0), "hyperplanes.pt") # save for next time
            print("Sex and Age coefficient loaded!")
    
    def __age__(self, w, step_size = -2, magnitude=1):
        alpha = step_size * magnitude
        return w + alpha * self.age_coeff
    
    def __sex__(self, w, step_size = 1, magnitude=1):
        alpha = step_size * magnitude
        return w + alpha * self.sex_coeff
    
    def augment_helper(self, embedding, rate=0.8): # p = augmentation rate
        np.random.seed(None); random.seed(None)
        if np.random.choice([True, False], p=[rate, 1-rate]): # random 80% chance of augmentation
            w_ = self.__sex__(embedding, magnitude=random.randint(-2,2))
            w_ = self.__age__(w_, magnitude=random.randint(-2,2))
            with torch.no_grad():
                synth, _ = self.generator([w_], input_is_latent=True)  # <-- Generate image here
            return synth
        return None
    
    def augment(self, sample, rate=0.8):
        with torch.no_grad():
            batch = self.encoder(torch.unsqueeze(sample, 0)) # sample patient
        batch = self.augment_helper(batch, rate)
        if batch is not None:
            # convert to (none, 224, 224, 3) numpy array
            batch = batch.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.float32).numpy()
            return np.array([cv2.resize(b, (224, 224)) for b in batch])
        return sample
        
    
    
# if __name__ == "__main__":
#     # initialize GCA
#     gca = GCA()
#     # load image 

#     # augment image with GCA
#     aug_x = gca.augment(embedding)
#     if aug_x is not None:
#         print("Augmented Image: ", aug_x.shape)
#     else:
#         print("Original Image: ", x.shape)
    
    
    
    
    

        

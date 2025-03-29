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

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
        self.ckpt = torch.load(self.ckpt, map_location=lambda storage, loc: storage) # load model checkpoint

class GCA():
    def __init__(self, distributed=False, h_path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
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
        if self.distributed: # use multiple gpus
            local_rank = int(os.environ["LOCAL_RANK"])
            self.generator = nn.parallel.DistributedDataParallel(
                generator,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
            self.encoder = nn.parallel.DistributedDataParallel(
                    encoder,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                )
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256,256)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
            ]
        )        
        # Get SVM coefficients
        self.sex_coeff, self.age_coeff = None, None
        self.__get_hyperplanes__()
        self.w_shape = None
        
        
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
            rsna_csv["Image Index"] = "../datasets/rsna/" + rsna_csv["Image Index"] # add prefix to path
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
            self.sex_coeff, self.age_coeff = hyperplanes[:512], hyperplanes[512:]
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
#         v = self.age_coeff.named_steps['linearsvc'].coef_[0].reshape((self.w_shape)) # get coefficients from hyperplane
#         v = (torch.from_numpy(v).float()).to(self.device)
        return w + alpha * self.age_coeff
    
    def __sex__(self, w, step_size = 1, magnitude=1):
        alpha = step_size * magnitude
#         v = self.age_coeff.named_steps['linearsvc'].coef_[0].reshape((self.w_shape)) # get coefficients from hyperplane
#         v = (torch.from_numpy(v).float()).to(self.device)
        return w + alpha * self.sex_coeff
    
    def augment_helper(self, embedding, rate=0.8): # p = augmentation rate
#         sex, age = gca.sex_coeff.predict(embedding.clone().detach().cpu().numpy())[0],\
#                     gca.age_coeff.predict(embedding.clone().detach().cpu().numpy())[0]
        np.random.seed(None); random.seed(None)
        if np.random.choice([True, False], p=[rate, 1-rate]): # random 80% chance of augmentation
            w_ = self.__sex__(embedding, magnitude=random.randint(-4,4))
            w_ = self.__age__(w_, magnitude=random.randint(-2,2))
#             if sex == "M":
#                 w_ = self.__sex__(embedding, magnitude=random.randint(-4,1))
#             else:
#                 w_ = self.__sex__(embedding, magnitude=random.randint(-1,4))
#             if age == "0-20": 
#                 w_ = self.__age__(w_, magnitude=random.randint(-1,4))
#             else:
#                 w_ = self.__age__(w_, magnitude=random.randint(-4,1))
            synth, _ = self.generator([w_], input_is_latent=True) # reconstruct image
            utils.save_image(synth, "real_samples_agesex.png", nrow=int(1 ** 2), normalize=True)
            return synth
#         synth, _ = self.generator([embedding], input_is_latent=True) # reconstruct image
        return None
    
    def augment(self, x, rate=0.8):
        x = torch.unsqueeze(self.transform(x), 0).to(self.device)
        embedding = self.encoder(x) # sample patient
        aug_x = self.augment_helper(embedding, rate)
        if aug_x is not None:
            # convert to (none, 224, 224, 3) numpy array
            im = utils.make_grid(aug_x)
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            return im.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = utils.make_grid(x)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        return im.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        
if __name__ == "__main__":
    # initialize GCA
    gca = GCA(h_path="hyperplanes.pt")
    # load image 
    img = cv2.imread("../datasets/rsna/00000007_000.png")
    gca.augment(img)
    
    
    # save or return image embedding
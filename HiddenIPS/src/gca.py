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

# args.cgan = True
build = {"G": True, "D": False, "E": True, "C": True}
size, n_mlp, channel_multiplier, cgan = 256, 8, 2, True
classifier_nof_classes, embedding_size, latent = 2, 10, 512
g_reg_every, lr, ckpt = 4, 0.002, 'results/000500.pt'
device = "cuda"


class GCA():
    def __init__(self, distributed=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.distributed = distributed
        self.size, self.n_mlp, self.channel_multiplier, self.cgan = 256, 8, 2, True
        self.classifier_nof_classes, self.embedding_size, self.latent = 2, 10, 512
        self.g_reg_every, self.lr, self.ckpt = 4, 0.002, 'results/000500.pt'
        
        self.generator = Generator(self.size, self.latent, self.n_mlp, channel_multiplier=self.channel_multiplier, 
                              conditional_gan=self.cgan, nof_classes=self.classifier_nof_classes, 
                              embedding_size=self.embedding_size).to(self.device)
        self.encoder = Encoder(size, channel_multiplier=self.channel_multiplier, output_channels=self.latent).to(self.device)
        self.generator.load_state_dict(ckpt["g"]); self.encoder.load_state_dict(ckpt["e"]) # load checkpoints
        if self.distributed: # use multiple gpus
            local_rank = int(os.environ["LOCAL_RANK"])
            generator = nn.parallel.DistributedDataParallel(
                generator,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
            encoder = nn.parallel.DistributedDataParallel(
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
        self.sex_coeff, self.age_coeff = get_hyperplanes()
    
    def initialize_models(self):
        def accumulate(model1, model2, decay=0.999):
            par1 = dict(model1.named_parameters())
            par2 = dict(model2.named_parameters())

            for k in par1.keys():
                par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
                self.ckpt = torch.load(self.ckpt, map_location=lambda storage, loc: storage) # load model checkpoint
        
    def load_image(self, path):
        img = cv2.imread(path)  # Load image using cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_tensor = transform(img_rgb).unsqueeze(0).to(device)  # Preprocess
        return img_tensor

    def process_in_batches(self, patients, encoder, batch_size):
        style_vectors = []
        for i in tqdm(range(0, len(patients), batch_size)):
            batch_paths = patients.iloc[i : i + batch_size]["Path"].tolist()
            batch_imgs = [self.load_image(path) for path in batch_paths]
            batch_imgs_tensor = torch.cat(batch_imgs, dim=0)  # Stack images in a batch
            with torch.no_grad():  # Avoid tracking gradients to save memory
                # Encode batch to latent vectors in Z space
                w_latents = encoder(batch_imgs_tensor)
            # Move to CPU to save memory and add to list
            style_vectors.extend(w_latents.cpu())
            del batch_imgs_tensor, w_latents # Cleanup and clear cache
            torch.cuda.empty_cache()  # Clear cache to free memory
        return style_vectors

    def load_cxr_data(self, df):
        return process_in_batches(df, encoder, batch_size=16)

    def get_patient_data(self, rsna_csv="../datasets/rsna_patients.csv", cxpt_csv="../chexpert/versions/1/train.csv"):
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

    def learn_linearSVM(self, d1, d2, df1, df2, key="Sex"):
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
        # Split dataset into train and test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(styles, labels, test_size=0.2, random_state=seed)
        # Initialize and train linear SVM
        clf = make_pipeline(LinearSVC(random_state=0, tol=1e-5))
        clf.fit(X_train, y_train)
        # Predict on the test set
        y_pred = clf.predict(X_test)
        return clf

    def get_hyperplanes(self):
        patient_data = self.get_patient_data()
        image_data = {}
        for key in patient_data:
            image_data[key] = load_cxr_data(patient_data[key])
        self.sex_coeff = self.learn_linearSVM(image_data["m"], image_data["f"], patient_data["m"], patient_data["f"])
        self.age_coeff = self.learn_linearSVM(image_data["y"], image_data["o"], patient_data["y"], patient_data["o"], key="Age")
        print("Sex and Age coefficient loaded!")
    
    def augment(self, img, p=0.8): # p = augmentation rate
        return False
    

        
if __name__ == "__main__":
    gca = GCA()
    

        

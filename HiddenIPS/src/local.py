'''
Configures model architecture
'''

import os
import numpy as np
import pandas as pd
from dataset import CustomDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import utils
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import random 
from stylegan2 import Generator, Encoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import torch.nn.functional as F
import time
import cProfile

'''
Multi-Attribute GCA
'''

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
        self.ckpt = torch.load(self.ckpt, map_location=lambda storage, loc: storage) # load model checkpoint

class GCA():
    def __init__(self, device="cuda", h_path = None):
        self.device = device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h_path = h_path # path to sex and age hyperplanes
        self.size, self.n_mlp, self.channel_multiplier, self.cgan = 256, 8, 2, True
        self.classifier_nof_classes, self.embedding_size, self.latent = 2, 10, 512
        self.g_reg_every, self.lr, self.ckpt = 4, 0.002, 'models/000500.pt'
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
        
    def __autoencoder__(self, img):
        x = self.encoder(img)
        synth, _ = self.generator([x], input_is_latent=True)
        return synth
        
    def reconstruct(self, img):
        return self.__autoencoder__(img)
        
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
        sample = sample.to(self.device)
        #sample = torch.unsqueeze(sample, 0)
        with torch.no_grad():
            batch = self.encoder(sample) # sample patient
        batch = self.augment_helper(batch, rate)
        if batch is not None:
            # convert to (none, 224, 224, 3) numpy array
            batch = batch.mul(255).add_(0.5).clamp_(0, 255)#.permute(0, 2, 3, 1)
            return F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
        return F.interpolate(sample, size=(224, 224), mode='bilinear', align_corners=False)
'''
Multi-Attribute GCA
'''


'''
Pneumonia Classifier
'''

class CustomModel(nn.Module):
    def __init__(self, base_model_name, num_classes=2):
        super(CustomModel, self).__init__()
        # Load the base model
        if base_model_name == 'densenet':
            self.base_model = models.densenet121(pretrained=True)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()  # Remove the original classifier
        elif base_model_name == 'resnet':
            self.base_model = models.resnet50(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the original classifier
        else:
            raise ValueError("Model not supported. Choose 'densenet' or 'resnet'")

        # Add custom classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(num_features, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        
        # Global average pooling
        if isinstance(x, torch.Tensor) and x.dim() == 4:  # Handle 4D tensor for CNNs
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Final classification layer
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def create_model(model_name, num_classes=2):
    return CustomModel(model_name, num_classes)

def create_dataloader(dataset, batch_size=32, shuffle=True, augmentation=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)# persistent_workers=True)
    return dataloader

'''
Local
'''

def __train_local(
    model_name,
    train_ds,
    val_ds,
    ckpt_dir,
    ckpt_name='model.pth',
    learning_rate=5e-5,
    epochs=100,
    image_shape=(224, 224, 3),
    augment=False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = create_model(model_name)
    model = model.to(device)
    if augment:
        gca = GCA(device=device, h_path='hyperplanes.pt')
        ckpt_dir = os.path.join("models/","GCA-"+ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        print("\nModel will be saved to: ", os.path.join(ckpt_dir, ckpt_name))
    else:
        gca = None
        ckpt_dir = os.path.join("models/",ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        print("\nModel will be saved to: ", os.path.join(ckpt_dir, ckpt_name))

    # Dataloaders
    train_ds = CustomDataset(csv_file=f'splits/trial_0/train.csv')
    train_loader = create_dataloader(train_ds, batch_size=64)
    val_loader = create_dataloader(val_ds, batch_size=64)  
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Since sigmoid is used, we use binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    logs = []
    # begin training
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training loop
        model.train()
        train_loss = 0.0
        all_labels, all_outputs = [], []
        
        with tqdm(train_loader, unit="batch", desc=f"Training Epoch {epoch + 1}/{epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device).float()
                if gca is not None:
                    images = gca.augment(images)
                outputs = model(images) # forward pass
                loss = criterion(outputs, labels)
                optimizer.zero_grad() # backpropagation
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_labels.extend(labels.cpu().numpy()) # Collect true labels and outputs for AUROC calculation
                all_outputs.extend(outputs.detach().cpu().numpy())
                # Calculate running AUROC (updated per batch)
                try:
                    batch_auc = roc_auc_score(np.array(all_labels), np.array(all_outputs), multi_class='ovr')
                except ValueError:
                    batch_auc = 0.0  # Handle potential errors in AUROC calculation (e.g., single class in batch)
                # Update pbar with current loss and AUROC
                pbar.set_postfix(loss=f"{loss.item():.4f}", auc=f"{batch_auc:.4f}")

        # Calculate epoch-level AUROC after all batches
        train_auc = roc_auc_score(np.array(all_labels), np.array(all_outputs), multi_class='ovr')
        
        # Validation loop
        model.eval()
        val_loss, val_labels, val_outputs = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Collect true labels and outputs for validation AUROC
                val_labels.extend(labels.cpu().numpy())
                val_outputs.extend(outputs.cpu().numpy())

        # Calculate validation AUROC
        val_auc = roc_auc_score(np.array(val_labels), np.array(val_outputs), multi_class='ovr')
        val_loss /= len(val_loader)

        # Display epoch summary
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} | Train AUROC: {train_auc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auc:.4f}"
        )


        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))

        # Log results
        logs.append([epoch + 1, train_loss, train_auc, val_loss, val_auc])
        
    # Explicit cleanup
    del train_loader
    torch.cuda.empty_cache()
    
    # Save logs
    logs_df = pd.DataFrame(logs, columns=['epoch', 'train_loss', 'train_auc', 'val_loss', 'val_auc'])
    logs_df.to_csv(os.path.join(ckpt_dir, f'{ckpt_name[:-4]}_logs.csv'), index=False)


def train_baseline(
    model_name,
    train_ds,
    val_ds,
    ckpt_dir,
    learning_rate=5e-5,
    epochs=100,
    image_shape=(224, 224, 3),
    augment=False
):
    __train_local(model_name, train_ds, val_ds, ckpt_dir, 'model.pth', learning_rate, epochs, image_shape, augment=augment)

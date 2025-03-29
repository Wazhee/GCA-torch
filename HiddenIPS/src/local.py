'''
Configures model architecture
'''

import os
import numpy as np
import pandas as pd
from dataset import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

'''
utils
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
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
):
    ckpt_dir = os.path.join("results/", ckpt_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ckpt_dir, exist_ok=True)
    print("\nModel will be saved to: ", os.path.join(ckpt_dir, ckpt_name))
    # Load model
    model = create_model(model_name)
    model = model.to(device)

    # Dataloaders
    train_loader = create_dataloader(train_ds)
    val_loader = create_dataloader(val_ds)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Since sigmoid is used, we use binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    logs = []
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training loop
        model.train()
        train_loss = 0.0
        all_labels, all_outputs = [], []
        
        with tqdm(train_loader, unit="batch", desc=f"Training Epoch {epoch + 1}/{epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device).float()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
             
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Collect true labels and outputs for AUROC calculation
                all_labels.extend(labels.cpu().numpy())
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
):
    __train_local(model_name, train_ds, val_ds, ckpt_dir, 'model.pth', learning_rate, epochs, image_shape)

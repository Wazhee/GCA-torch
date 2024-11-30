import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Custom Dataset for CheXpert
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.labels_df.iloc[idx, 0] # First column contains image paths
        image = Image.open(img_name).convert("RGB")  # Convert to RGB if grayscale
        label = 0 if self.labels_df.iloc[idx, 1] == "Female" else 1  # "sex" is the second column

        if self.transform:
            image = self.transform(image)

        return image, label

# Define paths
train_csv, train_dir = "../chexpert/versions/1/train.csv", "../chexpert/versions/1/train/"
test_csv, test_dir  = "../chexpert/versions/1/valid.csv", "../chexpert/versions/1/valid/"

# Image transformations (augmentation and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pretrained ResNet normalization
])

# Create datasets
loaded=False
try:
    train_dataset = CheXpertDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    test_dataset = CheXpertDataset(csv_file=test_csv, root_dir=test_dir, transform=transform)
    print("Chexpert Dataset Loaded...")
    loaded=True
except:
    print("Failed to load Chexpert dataset...")

# Running Trainig Loop
if(loaded):
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load the ResNet50 model
    model = torchvision.models.resnet50(pretrained=True)

    # Parallelize training across multiple GPUs
    model = torch.nn.DataParallel(model)

    # Set the model to run on the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get predictions
            correct_predictions += (predicted == labels).sum().item()  # Compare predictions to labels
            total_samples += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        # Print loss and accuracy for the epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    print('Finished Training!')
    
    # Save the model weights
    save_path = "../resnet50_gender_classifier.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
else:
    print("Exiting training loop...")

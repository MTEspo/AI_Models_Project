import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

# Set this to True if you want to see the features being extracted and False if you want to use the features
# that were all ready extracted
SHOW_FEATURES = False

# Set to True to load a saved model and False to train the MLP
REUSE_MLP = True

# Define transformations: convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to tensor
])


# Load the dataset, with download originally being set to true
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

print("Training dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

# Uncomment below if you want to see 5 of the pictures from the data that was extracted
'''Show a few images from the dataset, just to show that the pictures were extracted
def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.permute(1, 2, 0))  # Rearrange dimensions for display
        axes[i].axis('off')
        axes[i].set_title(f"Class: {label}")
    plt.show()

# Display 5 images from the training dataset
show_images(train_dataset)'''

print("Start by dividing the data into subsets and applying the transform to these subsets")
# Define transformations: Resize to 224x224, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean for each channel (ImageNet values)
        std=[0.229, 0.224, 0.225]  # Standard deviation for each channel
    )
])

# Subset the dataset to have 500 training images and 100 test images per class
train_subset = []
test_subset = []
class_counts_train = [0] * 10
class_counts_test = [0] * 10

# Subset the training dataset
for img, label in torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform):
    if class_counts_train[label] < 500:
        train_subset.append((img, label))
        class_counts_train[label] += 1

# Subset the testing dataset
for img, label in torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform):
    if class_counts_test[label] < 100:
        test_subset.append((img, label))
        class_counts_test[label] += 1

# Final Subset Sizes
print("Training subset size:", len(train_subset))  # Should show 5000
print("Test subset size:", len(test_subset))  # Should show 1000

# Create a DataLoader for batch processing
batch_size = 64  # Adjust based on memory
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


# Function for batch-wise feature extraction


def extract_features_batch(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting features"):
            batch_features = model(imgs).squeeze()  # Extract features
            features.append(batch_features)
            labels.append(lbls)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


if SHOW_FEATURES:
    # Load ResNet-18 with the latest weight handling
    weights = ResNet18_Weights.DEFAULT
    resnet18 = resnet18(weights=weights)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  # Remove the last layer
    resnet18.eval()  # Set model to evaluation mode

    # Extract features for training and testing datasets
    train_features, train_labels = extract_features_batch(train_loader, resnet18)
    test_features, test_labels = extract_features_batch(test_loader, resnet18)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    print("Feature extraction complete!")

    # Apply PCA to reduce dimensions from 512 to 50
    pca = PCA(n_components=50)

    # Fit PCA on training features
    train_features_reduced = pca.fit_transform(train_features)

    # Transform test features using the same PCA model
    test_features_reduced = pca.transform(test_features)

    print("Dimensionality reduction complete!")
    print(f"Reduced training feature shape: {train_features_reduced.shape}")
    print(f"Reduced testing feature shape: {test_features_reduced.shape}")

else:
    if os.path.exists("saved_features/train_features.npz") and os.path.exists("saved_features/test_features.npz"):
        # Load saved features
        train_data = np.load("saved_features/train_features.npz")
        train_features_reduced = train_data["features"]  # input data (x)
        train_labels = train_data["labels"]  # labels (y)

        test_data = np.load("saved_features/test_features.npz")
        test_features_reduced = test_data["features"]  # input data (x)
        test_labels = test_data["labels"]  # labels (y)

        print("Features loaded from saved files.")
    else:
        # Load ResNet-18 with the latest weight handling
        weights = ResNet18_Weights.DEFAULT
        resnet18 = resnet18(weights=weights)
        resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  # Remove the last layer
        resnet18.eval()  # Set model to evaluation mode

        # Extract features for training and testing datasets
        train_features, train_labels = extract_features_batch(train_loader, resnet18)
        test_features, test_labels = extract_features_batch(test_loader, resnet18)

        print(f"Number of batches in train_loader: {len(train_loader)}")
        print(f"Number of batches in test_loader: {len(test_loader)}")
        print("Feature extraction complete!")

        # Apply PCA to reduce dimensions from 512 to 50
        pca = PCA(n_components=50)
        train_features_reduced = pca.fit_transform(train_features)
        test_features_reduced = pca.transform(test_features)

        print("Dimensionality reduction complete!")
        print(f"Reduced training feature shape: {train_features_reduced.shape}")
        print(f"Reduced testing feature shape: {test_features_reduced.shape}")

        # Save features to disk
        np.savez("saved_features/train_features.npz", features=train_features_reduced, labels=train_labels)
        np.savez("saved_features/test_features.npz", features=test_features_reduced, labels=test_labels)
        print("Features saved to disk.")


# FROM HERE, THE MLP ALGORITHM STARTS

# Define the Multi-Layer Perceptron (MLP) with specified architecture
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(50, 512)  # Original Hidden Layer 1: Linear(50, 512)
        self.relu1 = torch.nn.ReLU()         # ReLU Activation
        self.fc2 = torch.nn.Linear(512, 512)  # Original Hidden Layer 2: Linear(512, 512)
        self.batchnorm = torch.nn.BatchNorm1d(512)  # Original BatchNorm(512)
        self.relu2 = torch.nn.ReLU()         # ReLU Activation

        # Comment out below to not use extra layer
        # self.fc_extra = torch.nn.Linear(512, 256)  # New Extra Layer: Linear(512, 256)
        # self.relu_extra = torch.nn.ReLU()  # ReLU Activation for Extra Layer
        # self.fc3extra = torch.nn.Linear(256, 10)  # Final Layer: Linear(256, 10)

        # Uncomment below to use original third layer
        self.fc3 = torch.nn.Linear(512, 10)  # Original Final Layer 3: Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.batchnorm(x)
        x = self.relu2(x)

        # Uncomment below to use extra layer
        # x = self.fc_extra(x)  # Pass through the extra layer
        # x = self.relu_extra(x)  # Apply activation
        # x = self.fc3extra(x)

        # Comment out below to use extra layer above
        x = self.fc3(x)
        return x  # Raw logits; Softmax is applied during evaluation with CrossEntropyLoss

# Constants
input_size = 50
num_classes = 10
learning_rate = 0.01  # Adjusted for SGD
momentum = 0.9
num_epochs = 10
batch_size = 128

# Prepare PyTorch DataLoaders for reduced datasets
train_dataset_reduced = torch.utils.data.TensorDataset(
    torch.tensor(train_features_reduced, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long)
)
test_dataset_reduced = torch.utils.data.TensorDataset(
    torch.tensor(test_features_reduced, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long)
)
train_loader_reduced = torch.utils.data.DataLoader(train_dataset_reduced, batch_size=batch_size, shuffle=True)
test_loader_reduced = torch.utils.data.DataLoader(test_dataset_reduced, batch_size=batch_size, shuffle=False)

# Instantiate the MLP model
mlp_model = MLP()

# Change the last word in the file path for different models
if os.path.exists("saved_models/mlp_model_base.pth") and REUSE_MLP:
    mlp_model.load_state_dict(torch.load("saved_models/mlp_model_base.pth", weights_only=True))
    print("MLP model loaded from file")

else:
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Multi-class classification loss
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=learning_rate, momentum=momentum)
    # Train the MLP model
    print("Training the MLP model...")
    mlp_model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_features, batch_labels in train_loader_reduced:
            # Forward pass
            outputs = mlp_model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(mlp_model.state_dict(), "saved_models/mlp_model_base.pth")
    print("MLP model saved to file")

# Evaluate the MLP model
print("Evaluating the MLP model...")
mlp_model.eval()  # Set model to evaluation mode
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader_reduced:
        outputs = mlp_model(batch_features)
        _, predicted = torch.max(outputs, 1)  # Predicted class is the index of max logit
        all_predictions.extend(predicted.numpy())
        all_labels.extend(batch_labels.numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"MLP Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm_mlp = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix (rows = true labels 0-9, columns = predictions):\n", cm_mlp)

# Display confusion matrix
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=[str(i) for i in range(num_classes)])
disp_mlp.plot(cmap='Blues')

# Classification report
report_mlp = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(num_classes)])
print("Classification Report:\n", report_mlp)

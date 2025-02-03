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
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

# Set this to True if you want to see the features being extracted and False if you want to use the features
# that were all ready extracted
SHOW_FEATURES = False

# Set this to True if you want the model to relearn the data and False if you want to load the model
RELEARN = False

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
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean for each channel (ImageNet values)
        std=[0.229, 0.224, 0.225]    # Standard deviation for each channel
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
print("Test subset size:", len(test_subset))      # Should show 1000

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
        train_features_reduced = train_data["features"]
        train_labels = train_data["labels"]

        test_data = np.load("saved_features/test_features.npz")
        test_features_reduced = test_data["features"]
        test_labels = test_data["labels"]

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


# FROM HERE, THE NAIVE BAYES ALGORITHM STARTS (first is from scratch, second is using scikit)


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)  # Unique class labels
        self.class_priors = {}       # P(C)
        self.means = {}              # Mean of each feature per class
        self.variances = {}          # Variance of each feature per class

        # Calculate priors, means, and variances for each class
        for c in self.classes:
            X_c = X[y == c]  # Subset of data for class c
            self.class_priors[c] = len(X_c) / len(X)  # P(C)
            self.means[c] = np.mean(X_c, axis=0)      # Feature means
            self.variances[c] = np.var(X_c, axis=0)   # Feature variances

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.class_priors[c])  # Log of prior P(C) (Uses log probabilities to avoid underflow)
            likelihood = np.sum(self._gaussian_likelihood(x, c))  # Log of P(X | C)
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]  # Class with highest posterior

    def _gaussian_likelihood(self, x, c):
        mean = self.means[c]
        var = self.variances[c]
        # Gaussian likelihood for each feature
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2 / (2 * var))


# Instantiate and train the (from-scratch) model
print("The Gaussian Naive-Bayes algorithm that was manually implemented:")

if RELEARN:
    model = GaussianNaiveBayes()
    model.fit(train_features_reduced, train_labels)
else:
    # Save the model or load the model if it was already saved before
    if os.path.exists("saved_models/gnb_from_scratch.pkl"):
        with open("saved_models/gnb_from_scratch.pkl", 'rb') as file:
            model = pickle.load(file)
        print("From-scratch model loaded.")
    else:
        model = GaussianNaiveBayes()
        model.fit(train_features_reduced, train_labels)
        with open("saved_models/gnb_from_scratch.pkl", 'wb') as file:
            pickle.dump(model, file)
        print("From-scratch model trained and saved.")

print("Model has been fit")
# Predict on the test set
predictions = model.predict(test_features_reduced)
print("Predictions have been set")

# Evaluate the model
print("Now evaluating the Gaussian Naive-Bayes algorithm that was manually implemented:")
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm_from_scratch = confusion_matrix(test_labels, predictions)
print("Confusion Matrix (rows = true labels 0-9, columns = predictions):\n", cm_from_scratch)
disp_from_scratch = ConfusionMatrixDisplay(confusion_matrix=cm_from_scratch, display_labels=np.unique(test_labels))
disp_from_scratch.plot(cmap='Blues')

report_from_scratch = classification_report(test_labels, predictions, target_names=[str(i) for i in range(10)])
print("Classification Report:\n", report_from_scratch)

# Now for the scikit GNB model
print("The SciKit Gaussian Naive-Bayes algorithm:")

if RELEARN:
    model_sklearn = GaussianNB()
    model_sklearn.fit(train_features_reduced, train_labels)
else:
    # Save the model or load the model if it was already saved before
    if os.path.exists("saved_models/gnb_scikit.pkl"):
        with open("saved_models/gnb_scikit.pkl", 'rb') as file:
            model_sklearn = pickle.load(file)
        print("Scikit model loaded.")
    else:
        model_sklearn = GaussianNB()
        model_sklearn.fit(train_features_reduced, train_labels)
        with open("saved_models/gnb_scikit.pkl", 'wb') as file:
            pickle.dump(model_sklearn, file)
        print("Scikit model trained and saved.")

print("Model has been fit")
# Test the model
predictions_sklearn = model_sklearn.predict(test_features_reduced)
print("Predictions have been set")

# Evaluate the model
print("Now evaluating the SciKit Gaussian Naive-Bayes algorithm:")
accuracy_sklearn = accuracy_score(test_labels, predictions_sklearn)
print(f"Scikit-learn GaussianNB Accuracy: {accuracy_sklearn * 100:.2f}%")

cm_sklearn = confusion_matrix(test_labels, predictions_sklearn)
print("Confusion Matrix (rows = true labels 0-9, columns = predictions):\n", cm_sklearn)
disp_sklearn = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn, display_labels=np.unique(test_labels))
disp_sklearn.plot(cmap='Blues')

report_sklearn = classification_report(test_labels, predictions_sklearn, target_names=[str(i) for i in range(10)])
print("Classification Report:\n", report_sklearn)

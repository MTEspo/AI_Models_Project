import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

# Set this to True if you want to see the features being extracted and False if you want to use the features
# that were all ready extracted
SHOW_FEATURES = False

# Set this to True to use the saved model and false to relearn the model
LOAD_MODEL = True

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


# FROM HERE, THE DECISION TREE ALGORITHM STARTS (first is from scratch, second is using scikit)


class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        # Start building the tree
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        # Predict for each sample in X
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _gini(self, y):
        # Compute Gini impurity for a set of labels
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _split(self, X, y, feature_index, threshold):
        # Split data into left and right subsets
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        # Find the best feature and threshold to split on
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        best_splits = None

        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Perform the split
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)

                # Compute the Gini impurity
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                # Update best split if this one is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold
                    best_splits = (X_left, X_right, y_left, y_right)

        return best_feature, best_threshold, best_splits

    def _build_tree(self, X, y, depth):
        # Stop conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            # Return the majority class
            return {'type': 'leaf', 'class': np.argmax(np.bincount(y))}

        # Find the best split
        feature, threshold, splits = self._best_split(X, y)

        if feature is None:
            # Return the majority class if no split is possible
            return {'type': 'leaf', 'class': np.argmax(np.bincount(y))}

        # Recursively build left and right subtrees
        X_left, X_right, y_left, y_right = splits
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
        }

    def _predict_single(self, x, tree):
        # Traverse the tree recursively for a single sample
        if tree['type'] == 'leaf':
            return tree['class']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


# Instantiate and train the (from-scratch) model
print("The Decision Tree algorithm that was manually implemented:")

# Change the last number in the file path to the desired max depth to get the proper model
if LOAD_MODEL and os.path.exists("saved_models/decision_tree_md50.pkl"):
    # Load the saved decision tree model
    with open("saved_models/decision_tree_md50.pkl", "rb") as f:
        tree = pickle.load(f)
    print("Loaded saved decision tree model")
else:
    # Instantiate and train the (from-scratch) decision tree model
    tree = DecisionTree(max_depth=50)
    tree.fit(train_features_reduced, train_labels)

    # Save the decision tree model
    with open("saved_models/decision_tree_md50.pkl", "wb") as f:
        pickle.dump(tree, f)
    print("Decision tree model saved")

print("Model has been fit")

# Predict on the test set
predictions = tree.predict(test_features_reduced)
print("Predictions have been set")

# Evaluate the model
print("Now evaluating the Decision Tree algorithm that was manually implemented:")
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(test_labels, predictions)
print("Confusion Matrix (rows = true labels 0-9, columns = predictions):\n", cm)

# Display confusion matrix
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(test_labels))
disp_dt.plot(cmap='Blues')

# Classification report
report = classification_report(test_labels, predictions, target_names=[str(i) for i in range(10)])
print("Classification Report:\n", report)

# Now for the scikit DT model
print("The SciKit Decision Tree algorithm:")

# Train the Decision Tree Classifier with Scikit-learn
print("The Decision Tree algorithm using Scikit-learn:")

# Change the last number in the file path to the desired max depth to get the proper model
if LOAD_MODEL and os.path.exists("saved_models/decision_tree_scikit_md50.pkl"):
    # Load the saved decision tree model
    with open("saved_models/decision_tree_scikit_md50.pkl", "rb") as f1:
        model_sklearn_dt = pickle.load(f1)
    print("Loaded saved decision tree model")
else:
    # Instantiate and train the (scikit) decision tree model
    model_sklearn_dt = DecisionTreeClassifier(max_depth=50, random_state=42)
    model_sklearn_dt.fit(train_features_reduced, train_labels)

    # Save the decision tree model
    with open("saved_models/decision_tree_scikit_md50.pkl", "wb") as f1:
        pickle.dump(model_sklearn_dt, f1)
    print("Decision tree model saved")

print("Model has been fit")

# Predict on the test set
predictions_sklearn_dt = model_sklearn_dt.predict(test_features_reduced)
print("Predictions have been set")

# Evaluate the Scikit-learn Decision Tree model
print("Now evaluating the Decision Tree algorithm using Scikit-learn:")
accuracy_sklearn_dt = accuracy_score(test_labels, predictions_sklearn_dt)
print(f"Accuracy: {accuracy_sklearn_dt * 100:.2f}%")

# Confusion Matrix
cm_sklearn_dt = confusion_matrix(test_labels, predictions_sklearn_dt)
print("Confusion Matrix (rows = true labels 0-9, columns = predictions):\n", cm_sklearn_dt)

# Display Confusion Matrix
disp_sklearn_dt = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn_dt, display_labels=np.unique(test_labels))
disp_sklearn_dt.plot(cmap='Blues')

# Classification Report
report_sklearn_dt = classification_report(test_labels, predictions_sklearn_dt, target_names=[str(i) for i in range(10)])
print("Classification Report:\n", report_sklearn_dt)

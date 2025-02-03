import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
import time  # For time measurement

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations: normalize and convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load the full CIFAR-10 dataset
train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# Subset logic for CIFAR-10: Use 500 training and 100 test images per class
train_subset, test_subset = [], []
class_counts_train = [0] * 10
class_counts_test = [0] * 10

# Subset the training dataset
for img, label in train_dataset_full:
    if class_counts_train[label] < 500:
        train_subset.append((img, label))
        class_counts_train[label] += 1

# Subset the testing dataset
for img, label in test_dataset_full:
    if class_counts_test[label] < 100:
        test_subset.append((img, label))
        class_counts_test[label] += 1

print("Training subset size:", len(train_subset))  # Should show 5000
print("Test subset size:", len(test_subset))      # Should show 1000

# Create DataLoaders for batch processing
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Define the VGG11 architecture with larger kernels
class VGG11_LargerKernels(nn.Module):
    def __init__(self):
        super(VGG11_LargerKernels, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # Conv1 (5x5 kernel)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # Conv2 (5x5 kernel)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),  # Conv3 (5x5 kernel)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),  # Conv4 (5x5 kernel)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool3

            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),  # Conv5 (5x5 kernel)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),  # Conv6 (5x5 kernel)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool4

            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),  # Conv7 (5x5 kernel)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),  # Conv8 (5x5 kernel)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),  # Adjust size based on input resolution
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)  # Output layer for 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize the modified VGG11 model
cnn_model = VGG11_LargerKernels().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

# Training loop with time and memory measurement
num_epochs = 10
cnn_model.train()

train_start_time = time.time()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_images, batch_labels in train_loader:
        # Move data to GPU
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        # Measure memory usage
        torch.cuda.reset_max_memory_allocated(device)

        # Forward pass
        outputs = cnn_model(batch_images)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track max memory used
        memory_used = torch.cuda.max_memory_allocated(device) / 1e6  # Convert bytes to MB
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Max Memory: {memory_used:.2f} MB")

train_end_time = time.time()
print(f"Training Time: {train_end_time - train_start_time:.2f} seconds")

# Save the trained model
torch.save(cnn_model.state_dict(), "CNN_larger_kernels.pth")
print("Model with larger kernels saved.")

# Load the saved model before evaluation
# cnn_model = VGG11_LargerKernels().to(device)  # Ensure to use the correct class name
# cnn_model.load_state_dict(torch.load("CNN_larger_kernels.pth"))  # Load the saved model
# cnn_model.eval()  # Set the model to evaluation mode
# print("Model with larger kernels loaded successfully.")



# Evaluation loop with time and memory measurement
cnn_model.eval()
all_predictions, all_labels = [], []
correct, total = 0, 0

eval_start_time = time.time()
torch.cuda.reset_max_memory_allocated(device)
with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        outputs = cnn_model(batch_images)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

        # Track memory during evaluation
        eval_memory_used = torch.cuda.max_memory_allocated(device) / 1e6  # Convert bytes to MB

eval_end_time = time.time()
print(f"Evaluation Time: {eval_end_time - eval_start_time:.2f} seconds")
print(f"Max Memory Used During Evaluation: {eval_memory_used:.2f} MB")

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[str(i) for i in range(10)]).plot(cmap='Blues')
plt.title("Confusion Matrix: VGG11 Larger Kernels")
plt.savefig("CNN_confusion_matrix_larger_kernels.png")
plt.show()

# Classification report
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
report = classification_report(all_labels, all_predictions, target_names=class_labels)
print("Classification Report:\n", report)

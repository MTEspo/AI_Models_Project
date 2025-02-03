import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations: normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# Subset logic for training and testing
train_subset = []
test_subset = []
class_counts_train = [0] * 10
class_counts_test = [0] * 10

# Limit to 500 training images and 100 test images per class
for img, label in torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform):
    if class_counts_train[label] < 500:
        train_subset.append((img, label))
        class_counts_train[label] += 1

for img, label in torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform):
    if class_counts_test[label] < 100:
        test_subset.append((img, label))
        class_counts_test[label] += 1

print("Training subset size:", len(train_subset))  # Should show 5000
print("Test subset size:", len(test_subset))      # Should show 1000

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Define a modified VGG11 model with three layers removed
class VGG11_RemoveThreeLayers(nn.Module):
    def __init__(self):
        super(VGG11_RemoveThreeLayers, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1
                
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Removed Conv4 and BatchNorm(256)
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool3
                
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Removed Conv6 and BatchNorm(512)
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool4
                
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Conv7
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Removed Conv8 and BatchNorm(512)
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool5
        )
        # Adjusted the feature size calculation (7x7 for 224x224 input after pooling)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),  # Input size adjusted for 32x32 images
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


# Initialize the modified model
cnn_model = VGG11_RemoveThreeLayers().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 10
cnn_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_images, batch_labels in train_loader:
        # Move data to device
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # Forward pass
        outputs = cnn_model(batch_images)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save the modified model
torch.save(cnn_model.state_dict(), "CNN_remove_three_layers.pth")
print("CNN model with three removed layers saved as CNN_remove_three_layers.pth")

# Load the saved model before evaluation
# cnn_model = VGG11_RemoveThreeLayers().to(device)
# cnn_model.load_state_dict(torch.load("CNN_remove_three_layers.pth"))
# cnn_model.eval()
# print("Model loaded successfully for evaluation.")

# Evaluation loop
cnn_model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for batch_images, batch_labels in test_loader:
        # Move data to device
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # Forward pass
        outputs = cnn_model(batch_images)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
# Print confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Display confusion matrix
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[str(i) for i in range(10)]).plot(cmap='Blues')
plt.title("Confusion Matrix: VGG11 Remove Three Layers")
plt.savefig("CNN_confusion_matrix_remove_3_layers.png")
plt.show()

# Classification report
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
report = classification_report(all_labels, all_predictions, target_names=class_labels)
print("Classification Report:\n", report)

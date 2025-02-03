import torchvision
import torchvision.transforms as transforms

# Define transformations: convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to tensor
])

# Load the dataset, with download originally being set to true
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("Training dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

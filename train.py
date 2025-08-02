import ssl
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError

# Fix SSL cert issue (Mac specific)
ssl._create_default_https_context = ssl._create_unverified_context

# Custom ImageFolder to skip bad images
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Skipping corrupted/unreadable image at index {index}: {e}")
                index = (index + 1) % len(self)

# Dataset directory
data_dir = 'data/train'

# Check PyTorch and torchvision versions
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# Hyperparameters
batch_size = 16
epochs = 2  # reduced for demo
lr = 1e-4

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = SafeImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained ResNet18 and modify final layer
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Classes:", dataset.classes)

# Training loop with accuracy
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate batch accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataset)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Save model and class names
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': dataset.classes
}, 'xray_bodypart_model.pth')

print("Training complete!")

import torch
from torchvision import transforms, models
from PIL import Image
import sys

# Load model and classes
checkpoint = torch.load('xray_bodypart_model.pth', map_location='cpu')
class_names = checkpoint['class_names']

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Image path from CLI or hardcoded
img_path = sys.argv[1] if len(sys.argv) > 1 else 'test_xray.jpg'

# Load and predict
try:
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted body part: {class_names[predicted.item()]}")
except Exception as e:
    print(f"Error: {e}")

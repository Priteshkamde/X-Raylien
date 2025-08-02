import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# Load model and class names
@st.cache_resource
def load_model():
    checkpoint = torch.load('xray_bodypart_model.pth', map_location='cpu')
    class_names = checkpoint['class_names']
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

model, class_names = load_model()

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🩻 X-Ray Body Part Classifier")
st.write("Upload an X-ray image and the model will predict which body part it is.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probs, 3)

        st.subheader("Top Predictions:")
        for i in range(top_probs.size(1)):
            st.write(f"**{class_names[top_indices[0][i]]}** — {top_probs[0][i].item() * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Could not process the image: {e}")

# X-Raylien

> [Blog Post](https://priteshkamde.github.io/medical-imaging/)
X-Raylien is a ResNet-based image classification model built to analyze X-ray images of **wrists**, **hands**, and **elbows**. Designed to assist in medical imaging research, the model classifies anatomical regions using high-quality radiographic data.

---

## Dataset

The project uses the **MURA** dataset.

📁 **Note**: The actual dataset files (`MURA-v1.1`, `MURA-processed`) are excluded from the repository using `.gitignore`. You can download and explore the dataset here:

Link [Dataset Reference](https://priteshkamde.github.io/medical-imaging/)

---

## Model Architecture

- **Backbone**: ResNet (pretrained on ImageNet)
- Trained with PyTorch

---

## Features

- Image preprocessing & augmentation
- Transfer learning with ResNet

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/X-Raylien.git
cd X-Raylien

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

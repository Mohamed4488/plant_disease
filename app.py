import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import os

model_path = "model/plant_disease_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

class_names = sorted(os.listdir("data/New Plant Diseases Dataset(Augmented)/train"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.set_page_config(page_title="Plant Disease Detector")
st.title("Plant Disease Prediction")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image,use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    predicted_class = class_names[pred_idx.item()]
    confidence_percent = confidence.item() * 100

    st.write(f"Predicted Disease: `{predicted_class}`")
    st.write(f"Accuarcy: `{confidence_percent:.2f}%`")
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Page setup ---
st.set_page_config(page_title="Food Freshness Classifier", page_icon="🍎")
st.title("🍎 Food Freshness & Quality Classifier")
st.write("Upload a photo of a fruit and get a visual freshness estimate.")

CLASS_NAMES = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
CONFIDENCE_THRESHOLD = 0.6

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/resnet50.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict(img):
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    if pred_class.startswith("fresh"):
        quality, food = "Fresh", pred_class.replace("fresh", "").capitalize()
    else:
        quality, food = "Rotten", pred_class.replace("rotten", "").capitalize()

    all_probs = {CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))}
    return food, quality, confidence, all_probs

# --- UI ---
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        food, quality, confidence, all_probs = predict(img)

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("Low-confidence prediction. Please upload a clearer image.")
        else:
            st.subheader(f"Food: {food}")
            if quality == "Fresh":
                st.success(f"Quality: {quality}")
            else:
                st.error(f"Quality: {quality}")
            st.write(f"Confidence: {confidence:.1%}")

            st.bar_chart(all_probs)

        st.caption("⚠️ This tool provides a visual estimate only and does not determine food safety, bacterial contamination, or microbial spoilage.")
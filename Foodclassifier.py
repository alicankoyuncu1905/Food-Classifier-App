import os
import uuid
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# 🚀 Streamlit config
st.set_page_config(page_title="🍣 Food Classifier", layout="centered")

# 📁 Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🏷️ Load class labels
@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# 🧠 Load TensorFlow model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("food_classifier_tensorflow.h5")

model = load_model()

# 🔍 Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension

    predictions = model.predict(image_array)[0]
    return {class_names[i]: round(float(pred) * 100, 2) for i, pred in enumerate(predictions)}

# 🌐 Streamlit UI
st.title("🍣 What Food Is This?")
st.write("Upload an image of **pizza**, **hamburger**, or **sushi** to find out what it is.")

uploaded_file = st.file_uploader("📤 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    filename = str(uuid.uuid4()) + ".jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="📸 Uploaded Image", use_container_width=True)

    predictions = predict(img_path)
    top_class = max(predictions, key=predictions.get)
    top_conf = predictions[top_class]

    st.subheader("🍽️ Result:")
    if top_conf >= 80.0:
        st.success(f"✅ It's **{top_class.capitalize()}**")
    else:
        st.warning("⚠️ Couldn't confidently recognize this food.")

    os.remove(img_path)

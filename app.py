import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌾🍅 Rice Tomato Detector")
st.write("Apne patte ki photo upload karo")

model = tf.keras.models.load_model("rice_tomato_model.h5")

uploaded = st.file_uploader("Photo yahan daalo", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Tumhari Photo", use_column_width=True)
    
    if st.button("Check Karo"):
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred)) * 100
        st.success(f"Result: Class {idx} - {conf:.2f}%")

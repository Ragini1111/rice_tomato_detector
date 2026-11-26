import streamlit as st
import numpy as np
from PIL import Image

# Tensorflow ko load karne me time lagta hai, isliye try-except me rakha hai
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# ================= PAGE SETTING =================
st.set_page_config(
    page_title="Rice Tomato Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Rice & Tomato Leaf Disease Detector")
st.write("Leaf ki photo upload karein, AI disease batayega + ilaj bhi batayega.")
st.markdown("---")

# ================= CLASS NAMES =================
# YE ORDER BAHUT IMPORTANT HAI
# Tumhare training wale order se match hona chahiye
CLASS_NAMES = [
    "Rice - Bacterial Blight",
    "Rice - Brown Spot",
    "Rice - Leaf Blast",
    "Rice - Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Bacterial Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Healthy"
]

# ================= DISEASE INFO =================
DISEASE_INFO = {
    "Rice - Bacterial Blight": {
        "symptoms": "Pattiyon ke kinare se peela hona aur sukhna, paani se bhige hue daag.",
        "prevention": "• Beej ko saaf paani se dho kar boyein\n• Khet me paani jama na hone dein\n• Nitrogen khaad zyada na dein",
        "treatment": "Streptocycline 0.15g + Copper Oxychloride 3g per litre paani me milakar chhidkav karein."
    },
    "Rice - Brown Spot": {
        "symptoms": "Pattiyon par bhoore gol daag, dana halka aur kharab.",
        "prevention": "• Swasth beej ka istemal karein\n• Khet me santulit khaad dein\n• Fasal chakra (Crop Rotation) apnayein",
        "treatment": "Mancozeb 2.5g per litre paani me milakar 10 din ke antar par chhidkav karein."
    },
    "Rice - Leaf Blast": {
        "symptoms": "Pattiyon par aankh jaisi - beech me grey, kinare brown daag.",
        "prevention": "• Nitrogen ki matra kam rakhein\n• Subah ke samay sinchai karein\n• Roggrast paudhe turant hata dein",
        "treatment": "Tricyclazole 0.6g per litre paani me milakar chhidkav karein."
    },
    "Rice - Healthy": {
        "symptoms": "Paudha bilkul swasth hai, koi rog nahi hai.",
        "prevention": "• Niyamit nirikshan karte rahein\n• Santulit sinchai aur khaad dete rahein",
        "treatment": "Koi dawai ki zaroorat nahi hai. Aise hi dekhbhal karte rahein."
    },
    "Tomato - Early Blight": {
        "symptoms": "Purani pattiyon par bhoore gol chhalle jaisi daag, peela pan.",
        "prevention": "• Paudhon ke beech doori rakhein\n• Neeche ki pattiyon ko mitti se touch na hone dein\n• Mulching karein",
        "treatment": "Chlorothalonil ya Mancozeb 2.5g per litre paani me milakar 7-10 din me chhidkav karein."
    },
    "Tomato - Late Blight": {
        "symptoms": "Pattiyon par kaale geele daag, jaldi pura paudha sukh jata hai.",
        "prevention": "• Zyada nami se bachayein\n• Roggrast paudhe turant ukhad kar jala dein\n• Barish ke baad turant dawa chhidkein",
        "treatment": "Metalaxyl + Mancozeb 2.5g per litre paani me chhidkav karein. Ye bahut tez failta hai, der na karein."
    },
    "Tomato - Leaf Mold": {
        "symptoms": "Pattiyon ke upar peele daag, neeche velvet jaisa fungus.",
        "prevention": "• Hawa ka aana-jaana achha rakhein\n• Pattiyon ko geela na rakhein, jad me paani dein",
        "treatment": "Chlorothalonil 2g per litre paani me chhidkav karein."
    },
    "Tomato - Bacterial Spot": {
        "symptoms": "Pattiyon aur fal par chhote kaale daag, patte jhadna.",
        "prevention": "• Beej ko garam paani se treat karein\n• Upar se paani na dein, drip irrigation karein",
        "treatment": "Copper Oxychloride 3g per litre paani me milakar chhidkav karein."
    },
    "Tomato - Yellow Leaf Curl Virus": {
        "symptoms": "Pattiyan mudna, peeli hona, paudha bauna reh jana.",
        "prevention": "• Safed makkhi (Whitefly) ko control karein\n• Peeli sticky trap lagayein\n• Roggrast paudha turant hata dein",
        "treatment": "Iska koi ilaaj nahi hai. Whitefly ke liye Imidacloprid ka chhidkav karein aur roggi paudha hata dein."
    },
    "Tomato - Healthy": {
        "symptoms": "Paudha bilkul swasth hai, koi rog nahi hai.",
        "prevention": "• Niyamit nirikshan karte rahein\n• Santulit khaad aur paani dete rahein",
        "treatment": "Koi dawai ki zaroorat nahi hai."
    }
}

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    # Yahan apne model ka naam likho - jaise model.h5 ya best_model.h5
    try:
        model = tf.keras.models.load_model("model.h5")
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Model file (model.h5) nahi mili. Apne model ka naam check karo - wo app.py ke saath wale folder me hona chahiye.")
    st.stop()

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Leaf ki photo yahan upload karein", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing... Please wait"):
            # Preprocess - 224x224 MobileNet ke liye
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            class_idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)

            disease_name = CLASS_NAMES[class_idx]
            info = DISEASE_INFO[disease_name]

            # ============ RESULT ============
            st.markdown("---")
            st.success(f"### ✅ Predicted Disease: {disease_name}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            if confidence < 70:
                st.warning("⚠️ Confidence kam hai. Kripya saaf photo dobara upload karein, achhi roshni me.")

            st.markdown("### 🩺 Symptoms:")
            st.write(info["symptoms"])

            st.markdown("### 🛡️ Preventive Measures:")
            st.write(info["prevention"])

            st.markdown("### 💊 Treatment / Upchar:")
            st.write(info["treatment"])

            st.markdown("---")
            st.caption("Note: Ye AI based suggestion hai, final diagnosis ke liye krishi expert se salah lein.")

else:
    st.info("👆 Pehle ek leaf ki photo upload karein.")

st.markdown("---")
//st.caption("Made by - Tiyasa, Swastika, Prapti, Ragini | Academy of Technology | Under Guidance of Prof. Biplab Ghosh")

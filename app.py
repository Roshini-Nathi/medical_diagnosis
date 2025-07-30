# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Page config
st.set_page_config(page_title="Chest X-ray Predictor", layout="centered")

# Simulated user database
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for MobileNetV2
    image = img_to_array(image)  # Convert to array
    image = preprocess_input(image)  # Preprocess for MobileNetV2
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Load model
@st.cache_resource
def load_cnn_model():
    return load_model("xray_model.h5")

model = load_cnn_model()

# User session check
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Page routing
if not st.session_state["logged_in"]:
    page = st.sidebar.selectbox("Select Page", ["Login", "Register"])
else:
    page = "Predict X-ray"

# Login page
if page == "Login":
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# Register page
elif page == "Register":
    st.title("📝 Register")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if new_username in st.session_state["users"]:
            st.warning("Username already exists.")
        else:
            st.session_state["users"][new_username] = new_password
            st.success("Registered successfully! Now login.")

# Prediction page
elif page == "Predict X-ray":
    st.title("🩺 Chest X-ray Prediction")
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        if st.button("Predict"):
            try:
                with st.spinner("Analyzing X-ray..."):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)[0][0]

                    label = "Pneumonia" if prediction > 0.5 else "Normal"
                    confidence = prediction if prediction > 0.5 else 1 - prediction

                    st.success(f"🩻 Prediction: *{label}*")
                    st.info(f"🧠 Confidence: {confidence * 100:.2f}%")

                    # Health Suggestions
                    if label == "Pneumonia":
                        st.warning("⚠ Health Advice:")
                        st.markdown("""
                        - Visit a pulmonologist ASAP.
                        - Take adequate rest and hydrate well.
                        - Use a humidifier if possible.
                        - Avoid crowded areas and wear a mask.
                        - Further tests: CBC, CRP, Chest CT recommended.
                        """)
                    else:
                        st.success("💡 Health Advice:")
                        st.markdown("""
                        - Lungs look healthy. Continue normal activities.
                        - Stay hydrated and eat a balanced diet.
                        - Avoid smoking or second-hand smoke.
                        - Maintain good air quality indoors.
                        - Include breathing exercises or yoga.
                        """)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
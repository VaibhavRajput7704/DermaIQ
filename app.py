# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import preprocess_image, predict_burn
from gradcam import preprocess_image, make_gradcam_heatmap, overlay_heatmap
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from streamlit_option_menu import option_menu

# ----------------- Streamlit Setup -----------------
st.set_page_config(page_title="DermaIQ", layout="centered", initial_sidebar_state="expanded")

# Inject custom CSS
st.markdown(f"""
    <style>
    .main {{
        background-color: #F3F6FB;
        color: white;
    }}
    .stApp {{
        font-family: 'Segoe UI', sans-serif;
    }}
    .title-text {{
        color: #4A90E2;
        font-size: 2.2rem;
        font-weight: 700;
    }}
    .subtitle-text {{
        color: white;
        font-size: 1.2rem;
    }}
    .gradcam-box {{
        border-radius: 10px;
        padding: 15px;
        background-color: #EAF1FB;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
        margin-top: 10px;
    }}
    </style>
""", unsafe_allow_html=True)


# ----------------- Sidebar -----------------
with st.sidebar:
    app_mode = option_menu(
        "DermaIQ",
        ["About", "Burn Detection", "AI Chat", "Dashboard"],
        icons=['house', 'fire', 'chat-dots', 'bar-chart'],
        menu_icon="cast",
        default_index=0,
        styles={
    "container": {"padding": "5px", "background-color": "#000000"},
    "icon": {"color": "#4A90E2", "font-size": "20px"},
    "nav-link": {
        "font-size": "16px",
        "text-align": "left",
        "margin": "5px",
        "--hover-color": "#4A90E2"
    },
    "nav-link-selected": {"background-color": "#ffffff", "color": "#000000"},
}

    )

# ----------------- Pages -----------------
if app_mode == "About":
    st.markdown('<div class="title-text">🏥 DermaIQ: Burn Grading and Treatment Recommendation Platform</div>', unsafe_allow_html=True)

    # Display image if exists
    st.image("image1.png", caption="DermaIQ System Overview", use_container_width=True)

    st.markdown("""
    <div class="subtitle-text">
    <b>DermaIQ</b> is a smart AI-powered diagnostic tool designed to help medical professionals and patients identify burn wound severity with the assistance of deep learning and explainable AI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ❓ Problem Statement")
    st.markdown("""
    Burn injuries are a major global health issue. Early and accurate classification into First, Second, or Third-degree burns is critical for treatment planning. Manual classification by visual inspection is often subjective and error-prone.
    """)

    st.markdown("### 🎯 Project Goal")
    st.markdown("""
    DermaIQ aims to automate burn classification using deep learning and provide treatment suggestions using an integrated medical AI chat system.
    """)

    st.markdown("### ⚙️ Tech Stack")
    st.markdown("""
    - **Frontend**: Streamlit
    - **Model**: TensorFlow/Keras (MobileNetV2 & Custom CNN)
    - **Explainability**: Grad-CAM
    - **AI Chat Assistant**: Gemini API
    - **Data Analysis**: Pandas, Seaborn, Matplotlib
    """)

    st.markdown("### 🔍 Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- 🔥 Burn Detection")
        st.markdown("- 📊 Model Dashboard")

    with col2:
        st.markdown("- 🧠 Model Selection")
        st.markdown("- 📸 Grad-CAM Visuals")

    with col3:
        st.markdown("- 💬 AI Medical Chat")
        st.markdown("- 🎯 Confidence Scores")

    st.markdown("### 🧾 Dataset Used")
    st.markdown("""
    - **Source**: [Kaggle Burn Dataset](https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset)
    - **Format**: Images with YOLO annotations (converted for classification)
    """)

    st.info("This project was built as part of a Machine Learning internship to demonstrate the integration of deep learning, explainable AI, and chat-based medical guidance.")


elif app_mode == "Burn Detection":
    st.markdown('<div class="title-text">🔥 Burn Detection with Model Selection</div>', unsafe_allow_html=True)
    st.markdown("Choose a model and upload an image to detect burn severity and visualize it using Grad-CAM.")

    model_choice = st.radio("Select Model", ["MobileNetV2", "Custom CNN"])
    uploaded_file = st.file_uploader("Upload a burn wound image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        burn_degrees = ["First-degree", "Second-degree", "Third-degree"]
        img_array = preprocess_image(image)

        if model_choice == "MobileNetV2":
            model = tf.keras.models.load_model("burn_model_final_legacy_fixed.h5")
            pred_class, confidence = predict_burn(model, image)
            last_conv_layer_name = "Conv_1"
        else:
            model = tf.keras.models.load_model("burn_model_customcnn_functional_legacy_fixed.h5")
            preds = model.predict(img_array)
            pred_class = int(np.argmax(preds))
            confidence = float(np.max(preds))
            last_conv_layer_name = "conv2d_14"

        st.success(f"🧠 Prediction: **{burn_degrees[pred_class]}**  \n📊 Confidence Score: **{confidence:.2f}**")

        st.markdown("#### 🔍 Grad-CAM Visualization")
        try:
            grad_model = tf.keras.models.Model(inputs=model.input,
                                               outputs=[model.get_layer(last_conv_layer_name).output, model.output])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = tf.argmax(predictions[0])
                class_output = predictions[:, pred_index]

            grads = tape.gradient(class_output, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

            gradcam_img = overlay_heatmap(image, heatmap)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(gradcam_img, caption="Grad-CAM Overlay", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Grad-CAM Error: {e}")

elif app_mode == "AI Chat":
    from gemini_chat_helper import init_chat
    st.markdown('<div class="title-text">💬 DermaIQ - AI Medical Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask me anything about skin burns – first aid, severity, treatment steps.")

    if st.button("🔁 Reset Chat"):
        st.session_state.gemini_chat = init_chat()
        st.session_state.chat_history = []

    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = init_chat()
        st.session_state.chat_history = []

    for user_msg, ai_msg in st.session_state.chat_history:
        st.chat_message("user").markdown(user_msg)
        st.chat_message("ai").markdown(ai_msg)

    user_input = st.chat_input("💬 Ask a question about burn care...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        try:
            response = st.session_state.gemini_chat.send_message(user_input)
            ai_reply = response.text.strip()
            st.chat_message("ai").markdown(ai_reply)
            st.session_state.chat_history.append((user_input, ai_reply))
        except Exception as e:
            st.error("❌ Error contacting Gemini API.")

elif app_mode == "Dashboard":
    st.markdown('<div class="title-text">📊 Model Evaluation Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # MobileNetV2
    with col1:
        st.subheader("MobileNetV2")
        try:
            with open("training_history.pkl", "rb") as f:
                history = pickle.load(f)

            fig1, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history["accuracy"], label="Train")
            ax[0].plot(history["val_accuracy"], label="Val")
            ax[0].set_title("Accuracy")
            ax[0].legend()

            ax[1].plot(history["loss"], label="Train")
            ax[1].plot(history["val_loss"], label="Val")
            ax[1].set_title("Loss")
            ax[1].legend()
            st.pyplot(fig1)

            y_true = np.load("y_true.npy")
            y_pred = np.load("y_pred_classes.npy")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["First", "Second", "Third"],
                        yticklabels=["First", "Second", "Third"],
                        ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
        except Exception as e:
            st.error(f"Error loading MobileNetV2 metrics: {e}")

    # Custom CNN
    with col2:
        st.subheader("Custom CNN")
        try:
            with open("training_history_customcnn.pkl", "rb") as f:
                history = pickle.load(f)

            fig2, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history["accuracy"], label="Train")
            ax[0].plot(history["val_accuracy"], label="Val")
            ax[0].set_title("Accuracy")
            ax[0].legend()

            ax[1].plot(history["loss"], label="Train")
            ax[1].plot(history["val_loss"], label="Val")
            ax[1].set_title("Loss")
            ax[1].legend()
            st.pyplot(fig2)

            y_true = np.load("y_true_customcnn.npy")
            y_pred = np.load("y_pred_customcnn.npy")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm2, ax_cm2 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                        xticklabels=["First", "Second", "Third"],
                        yticklabels=["First", "Second", "Third"],
                        ax=ax_cm2)
            ax_cm2.set_xlabel("Predicted")
            ax_cm2.set_ylabel("Actual")
            st.pyplot(fig_cm2)
        except Exception as e:
            st.error(f"Error loading Custom CNN metrics: {e}")

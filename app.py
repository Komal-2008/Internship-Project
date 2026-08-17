import streamlit as st
import numpy as np
from PIL import Image
import pickle
import io
import zipfile
from datetime import datetime
 
# ==========================================================
# PAGE CONFIG (must be the first Streamlit command)
# ==========================================================
st.set_page_config(
    page_title="Smart Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# NOTE: All styling below uses 100% native Streamlit components
# (containers, columns, metrics, dividers, badges via st.success/info,
# emojis) — no custom HTML/CSS injection. The color theme (green,
# professional look) comes from .streamlit/config.toml, which must
# sit next to this file for the theme to apply.
#
# Anchor link icons (the small 🔗 that appears on hover next to
# headings) are removed the native way: st.title()/st.subheader()
# use anchor=False, and section labels use **bold** markdown text
# instead of "###" headings (only real headings get anchor icons).
 
# ==========================================================
# LOAD MODEL AND CLASS NAMES (cached so it loads only once)
# ==========================================================
@st.cache_resource
def load_trained_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("smart_waste_classifier.keras")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names
 
# ==========================================================
# SESSION STATE INIT
# (keeps prediction results alive across reruns — e.g. when a
#  download button is clicked, Streamlit reruns the script, and
#  without session_state the result would disappear)
# ==========================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None      # dict: class, confidence, timestamp
if "pred_image_bytes" not in st.session_state:
    st.session_state.pred_image_bytes = None
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
 
# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
with st.sidebar:
    st.title("♻️ Smart Waste", anchor=False)
    st.subheader("Classification System", anchor=False)
    st.caption("AI Powered Waste Detection")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "📷 Upload & Classify", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.success("Let's build a cleaner and greener tomorrow 🌱")
 
# ==========================================================
# PAGE 1: HOME
# ==========================================================
if page == "🏠 Home":
    st.title("♻️ Smart Waste Classification System", anchor=False)
    st.subheader("AI-powered solution to classify waste and support a cleaner, greener environment", anchor=False)
    st.divider()
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Model Accuracy", "92%", "Trained model accuracy")
    with col2:
        st.metric("📦 Waste Categories", "10", "Different waste types")
    with col3:
        st.metric("🖼️ Dataset Images", "4,752")
 
    st.divider()
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        with st.container(border=True):
            st.subheader("⭐ Project Highlights", anchor=False)
            st.markdown("🤖 **AI-Powered Classification** — Uses Deep Learning (CNN) to classify waste accurately")
            st.markdown("📦 **10 Waste Categories** — Cardboard, E-Waste, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation")
            st.markdown("⚡ **Fast Prediction** — Provides quick and reliable classification results")
            st.markdown("📊 **Supports Recycling** — Helps users segregate waste correctly for better recycling")
            st.markdown("🌱 **Eco-Friendly Solution** — Encourages sustainable waste management")
 
    with col_b:
        with st.container(border=True):
            st.subheader("📖 How to Use", anchor=False)
            st.markdown("1️⃣ **Go to Upload & Classify** — Navigate to the Upload & Classify page")
            st.markdown("2️⃣ **Upload a Waste Image** — Drag & drop or browse a waste image")
            st.markdown("3️⃣ **Click Predict** — Click the Predict button to classify the waste")
            st.markdown("4️⃣ **View Predicted Category** — See the predicted waste category")
            st.markdown("5️⃣ **Download Result** — Get the report + image together in one ZIP")
 
    st.divider()
    st.warning(
        "⚠️ Disclaimer: This application is developed for educational purposes only. "
        "It should not be used as a substitute for professional advice."
    )
 
# ==========================================================
# PAGE 2: UPLOAD & CLASSIFY
# ==========================================================
 
elif page == "📷 Upload & Classify":
    st.title("📷 Upload & Classify", anchor=False)
    st.caption("Upload a waste image (drag & drop supported) and click Predict to know the type of waste.")
    st.divider()
 
    col_left, col_right = st.columns(2)
 
    with col_left:
        with st.container(border=True):
            st.subheader("📤 Upload Image", anchor=False)
            uploaded_file = st.file_uploader(
                "Drag & Drop an image here or click to browse",
                type=["jpg", "jpeg", "png"],
                help="Supports: JPG, JPEG, PNG | Max file size: 10MB"
            )
 
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
 
                # Reset stale prediction only when a NEW file is uploaded
                if st.session_state.last_uploaded_name != uploaded_file.name:
                    st.session_state.prediction = None
                    st.session_state.pred_image_bytes = None
                    st.session_state.last_uploaded_name = uploaded_file.name
 
                st.subheader("Image Preview", anchor=False)
                st.image(image, use_container_width=True)
 
            # Once a prediction exists for the currently uploaded image,
            # disable the Predict button so it can't be clicked again
            # until a new image is uploaded.
            already_predicted = st.session_state.prediction is not None
 
            predict_clicked = st.button(
                "🔍 Predict",
                use_container_width=True,
                type="primary",
                disabled=already_predicted,
            )
 
            if already_predicted:
                st.caption("✅ Prediction already done for this image. Upload a new image to predict again.")
 
            if predict_clicked:
                if image is None:
                    st.error("Please upload an image first.")
                else:
                    with st.spinner("Classifying..."):
                        model, class_names = load_trained_model()
 
                        # Preprocess image exactly like training (224x224, no rescale
                        # because the Rescaling layer is inside the model itself)
                        img_resized = image.resize((224, 224))
                        img_array = np.array(img_resized).astype("float32")
                        img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
 
                        predictions = model.predict(img_array)
                        predicted_index = np.argmax(predictions[0])
                        predicted_class = class_names[predicted_index]
                        confidence = float(np.max(predictions[0])) * 100
 
                    # Store the image as PNG bytes so it survives future reruns
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format="PNG")
 
                    st.session_state.prediction = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state.pred_image_bytes = img_buffer.getvalue()
 
                    # Rerun immediately so the button re-renders as disabled
                    st.rerun()
 
    with col_right:
        with st.container(border=True):
            st.subheader("🎯 Prediction Result", anchor=False)
 
            result = st.session_state.prediction
 
            if result is None:
                st.info("Upload an image and click **Predict** to see the result here.")
            else:
                predicted_class = result["class"]
                confidence = result["confidence"]
 
                st.markdown("**Type of Waste**")
                st.markdown(f"**🟢 {predicted_class}**")
                st.markdown("**Confidence Level**")
                st.progress(min(int(confidence), 100))
                st.markdown(f"**{confidence:.2f}%**")
 
                if confidence >= 80:
                    st.success(f"This is a **{predicted_class}** waste. High confidence prediction. ✅")
                elif confidence >= 50:
                    st.info(f"This is likely a **{predicted_class}** waste. Moderate confidence. ℹ️")
                else:
                    st.warning(f"This might be a **{predicted_class}** waste, but confidence is low. ⚠️")
 
                st.divider()
 
                # ---------------- Combined single download (ZIP) ----------------
                report_text = (
                    "Smart Waste Classification System - Result\n"
                    "--------------------------------------------\n"
                    f"Predicted Waste Category : {predicted_class}\n"
                    f"Confidence Level          : {confidence:.2f}%\n"
                    f"Generated On              : {result['timestamp']}\n"
                )
 
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("waste_prediction_result.txt", report_text)
                    if st.session_state.pred_image_bytes:
                        zf.writestr("uploaded_waste_image.png", st.session_state.pred_image_bytes)
                zip_buffer.seek(0)
 
                # Build a unique filename using date & time so the browser
                # never has to append (1), (2), etc. on repeated downloads.
                file_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                zip_filename = f"waste_prediction_result_{file_timestamp}.zip"
 
                st.download_button(
                    label="⬇️ Download Result (Report + Image, ZIP)",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip",
                    use_container_width=True,
                )
 
# ==========================================================
# PAGE 3: ABOUT
# ==========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project", anchor=False)
    st.caption("Learn more about the model, dataset, and technologies behind this app.")
    st.divider()
 
    st.info(
        "**Smart Waste Classification System** is an AI-based web application that helps "
        "users identify and classify different types of waste using Deep Learning (CNN). "
        "The system accepts an image of waste and predicts its category instantly, "
        "helping in proper waste segregation and recycling."
    )
 
    st.subheader("🗂️ Waste Categories (Classes in Dataset)", anchor=False)
 
    categories = [
        ("📦", "Cardboard"), ("💻", "E-Waste"), ("🍌", "Food Organics"),
        ("🍾", "Glass"), ("🥫", "Metal"), ("🗑️", "Miscellaneous Trash"),
        ("📄", "Paper"), ("🧴", "Plastic"), ("👕", "Textile Trash"), ("🌿", "Vegetation")
    ]
 
    cols = st.columns(5)
    for i, (emoji, name) in enumerate(categories):
        with cols[i % 5]:
            with st.container(border=True):
                st.subheader(emoji, anchor=False)
                st.markdown(f"**{i+1}. {name}**")
 
    st.divider()
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        with st.container(border=True):
            st.subheader("🎯 Project Objective", anchor=False)
            st.markdown("- Build an AI model to classify waste into multiple categories")
            st.markdown("- Provide a simple and easy to use web interface")
            st.markdown("- Promote waste segregation and recycling")
            st.markdown("- Support a cleaner and greener environment")
 
    with col2:
        with st.container(border=True):
            st.subheader("🗄️ Dataset Information", anchor=False)
            st.markdown("- **Total Classes** : 10")
            st.markdown("- **Total Images** : 4,752")
            st.markdown("- **Source** : Local Dataset (RealWaste)")
 
    with col3:
        with st.container(border=True):
            st.subheader("💻 Technologies Used", anchor=False)
            st.markdown("- Python")
            st.markdown("- Streamlit")
            st.markdown("- TensorFlow / Keras")
            st.markdown("- CNN")
            st.markdown("- NumPy / Pillow")
 
    st.divider()
    st.warning(
        "⚠️ Disclaimer: This application is developed for educational purposes only. "
        "It should not be used as a substitute for professional advice."
    )
  
import streamlit as st
import cv2
import numpy as np
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from fpdf import FPDF
from pathlib import Path
import logging
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeukemiaDetector:
    CLASS_LABELS = {
        'Negative': {
            'description': "No leukemia detected",
            'color': "green",
            'details': "Regular checkup recommended"
        },
        'Early Level': {
            'description': "Early stage leukemia",
            'color': "orange",
            'details': "Consult hematologist"
        },
        'Pre Level': {
            'description': "Advanced pre-leukemia",
            'color': "#FF6666",
            'details': "Immediate consultation required"
        },
        'Pro Level': {
            'description': "Progressive leukemia",
            'color': "red",
            'details': "Emergency medical attention needed"
        }
    }
    
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        try:
            return load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def predict(self, image):
        try:
            resized = cv2.resize(image, (224, 224))
            preprocessed = preprocess_input(np.expand_dims(resized, axis=0))
            prediction = self.model.predict(preprocessed, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            labels = list(self.CLASS_LABELS.keys())
            return {
                'stage': labels[predicted_class_idx],
                'info': self.CLASS_LABELS[labels[predicted_class_idx]]
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

class ReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
        
    def generate_report(self, diagnosis_data, original_img):
        try:
            self.pdf.add_page()
            self.pdf.set_font("Arial", 'B', 16)
            self.pdf.cell(0, 10, "Leukemia Detection Report", ln=True, align='C')
            self.pdf.set_font("Arial", '', 10)
            self.pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
            self.pdf.set_font("Arial", 'B', 12)
            self.pdf.cell(0, 10, f"Stage: {diagnosis_data['stage']}", ln=True)
            self.pdf.set_font("Arial", '', 12)
            self.pdf.cell(0, 10, f"Description: {diagnosis_data['info']['description']}", ln=True)
            self.pdf.cell(0, 10, f"Recommendation: {diagnosis_data['info']['details']}", ln=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                temp_img_path = temp_img.name
                cv2.imwrite(temp_img_path, cv2.resize(original_img, (100, 100)))
            
            self.pdf.image(temp_img_path, x=10, y=None, w=50)
            os.remove(temp_img_path)
            
            report_bytes = self.pdf.output(dest='S').encode('latin1')
            return report_bytes
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise

def get_base64(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def setup_page():
    st.set_page_config(page_title="Leukemia Detection", layout="wide", page_icon="🩸")
    
    background_image = "C://Users//vinoth//Desktop//Main project//main-project//images//background.webp"  # Make sure the file exists
    if not os.path.exists(background_image):
        st.error("Background image not found!")
        return

    bg_base64 = get_base64(background_image)
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{bg_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .block-container {{
            padding: 2rem;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_page()
    
    try:
        model_path = "models.keras"
        detector = LeukemiaDetector(model_path)
    except Exception as e:
        st.error(f"Failed to initialize detector: {e}")
        return
    
    st.markdown("<h1 style='text-align: center; color: #ff4444;'>LEUKEMIA DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Blood Sample Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("Invalid image. Please upload a valid blood sample image.")
                return

            img = cv2.resize(img, (150, 150))
            st.image(img, caption="Blood Sample", width=150)

            with st.spinner("Analyzing..."):
                diagnosis = detector.predict(img)

            st.markdown(f"""
                <div style='background-color: {diagnosis["info"]["color"]}; padding: 15px; border-radius: 10px; color: white;'>
                    <h3>Detection Result: {diagnosis["stage"]}</h3>
                    <p>{diagnosis["info"]["description"]}</p>
                    <p><strong>Recommendation:</strong> {diagnosis["info"]["details"]}</p>
                </div>
            """, unsafe_allow_html=True)

            report_generator = ReportGenerator()
            report_bytes = report_generator.generate_report(diagnosis, img)
            
            st.download_button("📄 Download Report", report_bytes, file_name="leukemia_report.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()

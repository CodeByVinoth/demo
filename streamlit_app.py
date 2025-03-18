import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from fpdf import FPDF
from pathlib import Path
import logging
from datetime import datetime

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
            # Resize to model input size (224x224)
            resized = cv2.resize(image, (224, 224))
            # Preprocess for model
            preprocessed = preprocess_input(np.expand_dims(resized, axis=0))
            # Make prediction
            prediction = self.model.predict(preprocessed, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            
            # Get corresponding label
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
            
            # Header
            self.pdf.set_font("Arial", 'B', 16)
            self.pdf.cell(0, 10, "Leukemia Detection Report", ln=True, align='C')
            
            # Date
            self.pdf.set_font("Arial", '', 10)
            self.pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
            
            # Results
            self.pdf.set_font("Arial", 'B', 12)
            self.pdf.cell(0, 10, f"Stage: {diagnosis_data['stage']}", ln=True)
            self.pdf.set_font("Arial", '', 12)
            self.pdf.cell(0, 10, f"Description: {diagnosis_data['info']['description']}", ln=True)
            self.pdf.cell(0, 10, f"Recommendation: {diagnosis_data['info']['details']}", ln=True)
            
            # Save image to PDF
            temp_img_path = "temp_image.jpg"
            cv2.imwrite(temp_img_path, cv2.resize(original_img, (100, 100)))
            self.pdf.image(temp_img_path, x=10, y=None, w=50)
            os.remove(temp_img_path)
            
            # Save report to bytes
            report_bytes = self.pdf.output(dest='S').encode('latin1')
            return report_bytes
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise

def setup_page():
    st.set_page_config(page_title="Leukemia Detection", layout="wide", page_icon="🩸")
    
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a0000 0%, #000000 100%);
            color: #ffffff;
        }
        .result-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .small-image {
            max-width: 150px;
            margin: auto;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_page()
    
    try:
        # Use environment variable or relative path for model
        model_path = os.getenv("MODEL_PATH", "models.keras")
        detector = LeukemiaDetector(model_path)
    except Exception as e:
        st.error(f"Failed to initialize detector: {e}")
        return
    
    st.markdown("""
        <h1 style='text-align: center; color: #ff4444; font-size: 2em; margin-bottom: 1em;'>
            LEUKEMIA DETECTION SYSTEM
        </h1>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Blood Sample Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (150, 150))  # Resize for display
            
            # Display image
            st.image(img, caption="Blood Sample", use_column_width=False, width=150)
            
            # Predict
            with st.spinner("Analyzing..."):
                diagnosis = detector.predict(img)
            
            # Display result
            st.markdown(f"""
                <div class='result-card' style='border-left: 5px solid {diagnosis["info"]["color"]};'>
                    <h3>Detection Result: {diagnosis["stage"]}</h3>
                    <p>{diagnosis["info"]["description"]}</p>
                    <p><strong>Recommendation:</strong> {diagnosis["info"]["details"]}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Generate and download report
            report_generator = ReportGenerator()
            report_bytes = report_generator.generate_report(diagnosis, img)
            
            st.download_button(
                label="📄 Download Report",
                data=report_bytes,
                file_name=f"leukemia_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
            
            if st.button("🔄 New Analysis"):
                st.session_state.clear()
                st.rerun()
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
import streamlit as st
import easyocr
import tempfile
import os
from PIL import Image
import google.generativeai as genai
import pandas as pd
import time

# Set page config
st.set_page_config(page_title="Lab Report AI Assistant", layout="centered")

# Enhanced Custom CSS with Animation & Modern UI
st.markdown("""
    <style>
        .stApp {
            animation: fadeIn 1s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        body, .stApp {
            background-color: #f2f6fc;
            color: #000000 !important;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3, h4, h5, h6, p, span, div {
            color: #000000 !important;
        }

        .stButton > button {
            background-color: #5c6bc0;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 10px rgba(92, 107, 192, 0.2);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #3949ab;
            transform: scale(1.02);
        }

        .stDownloadButton > button {
            background-color: #26a69a;
            color: white;
            font-weight: bold;
            border-radius: 20px;
            padding: 0.5em 1em;
            box-shadow: 0 3px 6px rgba(0, 150, 136, 0.3);
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

        .stDownloadButton > button:hover {
            background-color: #00897b;
            transform: scale(1.03);
        }

        .stTextInput input, .stTextArea textarea {
            border-radius: 8px;
            padding: 0.4em 0.6em;
            border: 1px solid #ccc;
            transition: box-shadow 0.3s ease;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            box-shadow: 0 0 8px rgba(92, 107, 192, 0.4);
        }

        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .stExpander {
            border: 1px solid #dcdcdc;
            border-radius: 12px;
            background-color: #ffffff;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .stSpinner {
            font-style: italic;
            color: #3949ab !important;
        }

        .stRadio > div {
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 AI-Powered Medical Lab Report Assistant")

# Upload
uploaded_file = st.file_uploader("📤 Upload your medical lab report (Image/PDF)", type=["png", "jpg", "jpeg", "pdf"])

# OCR Function
def extract_text_from_image(file):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(file)
    extracted_text = " ".join([entry[1] for entry in result])
    return extracted_text

# Gemini Configuration
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-pro")

# Process
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    with st.spinner("🔍 Extracting text from your report..."):
        if file_extension == ".pdf":
            extracted_text = "PDF OCR not implemented in this version"
        else:
            extracted_text = extract_text_from_image(temp_path)

    with st.expander("📄 Extracted Text"):
        st.write(extracted_text)

    # AI Analysis
    if extracted_text:
        st.subheader("💡 AI Analysis & Explanation")

        with st.spinner("🧠 Analyzing with AI..."):
            prompt = f"""
            You are a medical AI assistant. Analyze the following lab report text. 
            Extract all key metrics, test results, and their normal ranges (if possible). 
            Explain any abnormal values clearly and concisely. Provide health advice if needed.

            Lab Report Text:
            {extracted_text}
            """
            try:
                response = model.generate_content(prompt)
                explanation = response.text
                st.success("✅ Analysis complete!")
                st.markdown(explanation)
            except Exception as e:
                st.error(f"Error from Gemini API: {e}")

import streamlit as st
import google.generativeai as genai
from PIL import Image
import PyPDF2
import tempfile
import os
from google.api_core import exceptions
from dotenv import load_dotenv
import time
from fpdf import FPDF
import io
import pandas as pd
import re

# Optional: Uncomment and install pytesseract if you want OCR from images
# import pytesseract

# Custom CSS for UI
st.markdown("""
    <style>
        /* Global text color and background */
        body, .stApp {
            background-color: #f2f6fc;
            color: #000000 !important;  /* Force black text */
            font-family: 'Segoe UI', sans-serif;
        }
        /* Other CSS styles... */
        .stButton > button {
            background-color: #5c6bc0;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #3949ab;
        }
        /* Add rest of your CSS here as needed */
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Helper functions

def parse_medical_report(text):
    """
    Parses medical test results from raw text.
    Returns list of dicts like:
    [{'test_name': 'Hemoglobin', 'value': 13.2, 'unit': 'g/dL', 'normal_range': (12.0, 16.0)}, ...]
    """
    results = []
    pattern = re.compile(r"(\w+(?: \w+)*)\s*:\s*([\d.]+)\s*(\w+)?\s*\(?([\d.]+)-([\d.]+)\)?", re.I)
    for match in pattern.finditer(text):
        test_name = match.group(1).strip()
        value = float(match.group(2))
        unit = match.group(3) if match.group(3) else ""
        normal_low = float(match.group(4))
        normal_high = float(match.group(5))
        results.append({
            'test_name': test_name,
            'value': value,
            'unit': unit,
            'normal_range': (normal_low, normal_high)
        })
    return results

def categorize_value(value, normal_range):
    low, high = normal_range
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    else:
        return "Normal"

def generate_explanation_for_test(model, test_name, value, unit, normal_range):
    """
    Generates explanation for test result using AI or placeholder.
    """
    low, high = normal_range
    if value < low:
        status = "below normal range"
    elif value > high:
        status = "above normal range"
    else:
        status = "within normal limits"
    # You can enhance this by calling the AI model if desired.
    return f"The {test_name} result is {value} {unit}, which is {status}."

def preprocess_image(image_path):
    """
    Preprocess image for OCR. Converts to grayscale for better OCR results.
    """
    image = Image.open(image_path)
    return image.convert("L")

def extract_text_from_image(image):
    """
    Extract text from image using OCR.
    Uncomment pytesseract import and install pytesseract to enable real OCR.
    """
    # Uncomment below if pytesseract is installed and imported
    # return pytesseract.image_to_string(image)

    # Placeholder example text
    return "Hemoglobin: 13.2 g/dL (12.0-16.0)\nWBC: 7000 /µL (4000-11000)"

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            st.warning("No text extracted from PDF. It may contain scanned images.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def analyze_medical_report(content, content_type):
    """
    Calls the Google Gemini AI to analyze medical report text or image content.
    """
    prompt = """
Analyze this medical report concisely. Provide:
1. Key findings and diagnoses (e.g., abnormal test results).
2. Specific dietary recommendations:
   - Foods to eat to address any abnormalities.
   - Foods to avoid to prevent worsening conditions.
3. General health actions (e.g., exercise, follow-up tests, lifestyle changes).
4. Any critical warnings or urgent actions needed.
If the report is unclear or empty, provide a fallback analysis with general health advice.
"""

    MAX_RETRIES = 3
    RETRY_DELAY = 2

    for attempt in range(MAX_RETRIES):
        try:
            if content_type == "image":
                # For image, send prompt + content as list
                response = model.generate_content([prompt, content])
            else:
                response = model.generate_content(f"{prompt}\n\n{content}")
            if not response.text.strip():
                st.warning("AI returned empty analysis.")
                return fallback_analysis(content, content_type)
            return response.text
        except exceptions.GoogleAPIError as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"API error. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)

def fallback_analysis(content, content_type):
    """
    Provides fallback general advice if AI analysis fails.
    """
    st.warning("Using fallback analysis due to API issues.")
    if content_type == "image":
        return """
Fallback Analysis:
- Unable to analyze the image due to API issues.
- Recommendations:
  - Consult a medical professional for accurate interpretation.
  - General Diet: Focus on a balanced diet with vegetables, lean proteins, and whole grains.
  - Avoid processed foods, sugary drinks, and excessive saturated fats.
  - Actions: Schedule a follow-up with a healthcare provider.
"""
    else:
        word_count = len(content.split())
        return f"""
Fallback Analysis:
- Document Type: Text-based medical report
- Word Count: Approximately {word_count} words
- Content: The document appears to contain medical information, but detailed analysis is unavailable.
- Recommendations:
  - Diet: Include fiber-rich foods (oats, vegetables), lean proteins (chicken, fish), and healthy fats (avocado, nuts).
  - Avoid: Sugary foods, trans fats, and excessive alcohol.
  - Actions: Consult a healthcare professional and consider regular exercise (30 min/day).
"""

def generate_pdf_report(text):
    """
    Generates a downloadable PDF report from the AI analysis text.
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'AI Medical Report Analysis', border=False, ln=True, align='C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary of Medical Report", ln=True, align='L')
    pdf.ln(4)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    cleaned_text = text.replace('\u200b', '').strip()
    paragraphs = cleaned_text.split('\n')
    for para in paragraphs:
        para = para.strip()
        if para:
            pdf.multi_cell(0, 10, f"    {para}")
            pdf.ln(2)
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 10)
    import datetime
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f'Report generated on: {gen_time}', ln=True, align='R')

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = io.BytesIO(pdf_bytes)
    pdf_buffer.seek(0)
    return pdf_buffer

def display_results_table(text):
    """
    Parses test results and displays them in a Streamlit dataframe.
    """
    test_results = parse_medical_report(text)
    if not test_results:
        st.warning("No test results were parsed from the report.")
        return []

    table_data = []
    for result in test_results:
        category = categorize_value(result['value'], result['normal_range'])
        ref_range = f"{result['normal_range'][0]} - {result['normal_range'][1]} {result['unit']}"
        table_data.append({
            "Test Name": result['test_name'],
            "Value": f"{result['value']} {result['unit']}",
            "Category": category,
            "Reference Range": ref_range
        })

    df = pd.DataFrame(table_data)
    st.subheader("📊 Extracted Test Results")
    st.dataframe(df, use_container_width=True)
    return test_results

def display_explanations(test_results, model):
    """
    For each test, display AI-generated explanations in expanders.
    """
    st.subheader("🧠 Explanations by AI")
    for result in test_results:
        with st.expander(f"{result['test_name']} Explanation"):
            explanation = generate_explanation_for_test(
                model,
                result['test_name'],
                result['value'],
                result['unit'],
                result['normal_range']
            )
            st.write(explanation)

# -----------------------
# Load environment and initialize Google Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.Models.get("gemini-1")

# -----------------------
# Streamlit UI

st.title("🧬 Medical Report Analyzer with AI Assistance")

# Upload file or paste text
input_option = st.radio("Select input type:", ("Upload medical report (PDF/Image)", "Paste report text"))

if input_option == "Upload medical report (PDF/Image)":
    uploaded_file = st.file_uploader("Upload a PDF or image file", type=['pdf', 'png', 'jpg', 'jpeg'])
    if uploaded_file:
        # Display preview
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            preprocessed_image = preprocess_image(uploaded_file)
            # OCR extraction (placeholder)
            extracted_text = extract_text_from_image(preprocessed_image)
            st.markdown("### Extracted Text from Image")
            st.text_area("Extracted Text:", extracted_text, height=200)
            input_text = extracted_text
            content_type = "image"
        elif uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text.strip():
                st.markdown("### Extracted Text from PDF")
                st.text_area("Extracted Text:", extracted_text, height=200)
                input_text = extracted_text
                content_type = "pdf"
            else:
                st.error("Could not extract text from the PDF file.")
                input_text = None
                content_type = None
        else:
            st.error("Unsupported file type.")
            input_text = None
            content_type = None

elif input_option == "Paste report text":
    input_text = st.text_area("Paste your medical report text here:", height=300)
    content_type = "text" if input_text else None

if input_text and content_type:
    if st.button("Analyze Medical Report"):
        with st.spinner("Analyzing report, please wait..."):
            ai_analysis = analyze_medical_report(input_text, content_type)
        st.markdown("## 📝 AI Analysis Result")
        st.write(ai_analysis)

        # Show test results in table and explanations if text content
        test_results = display_results_table(input_text)
        if test_results:
            display_explanations(test_results, model)

        # Offer PDF download of AI analysis
        pdf_buffer = generate_pdf_report(ai_analysis)
        st.download_button(
            label="Download AI Analysis as PDF",
            data=pdf_buffer,
            file_name="medical_report_analysis.pdf",
            mime="application/pdf"
        )

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

# Custom CSS
st.markdown("""
    <style>
        /* Global text color and background */
        body, .stApp {
            background-color: #f2f6fc;
            color: #000000 !important;  /* Force black text */
            font-family: 'Segoe UI', sans-serif;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
        }

        /* Paragraphs and spans */
        p, span, div {
            color: #000000 !important;
        }

        /* Streamlit elements */
        .stMarkdown, .stText, .stDataFrame, .stExpanderContent {
            color: #000000 !important;
        }

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

        .stRadio > div {
            padding: 10px 0px;
        }

        .css-1v0mbdj p {
            color: #000000 !important;
        }

        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            color: #000000 !important;
        }

        .stExpander {
            border: 1px solid #dcdcdc;
            border-radius: 10px;
            background-color: #ffffff;
            color: #000000 !important;
        }

        .stTextArea, .stTextInput {
            border-radius: 8px !important;
            color: #000000 !important;
        }

        .stDownloadButton > button {
            background-color: #26a69a;
            color: white;
            font-weight: bold;
            border-radius: 20px;
        }

        .stDownloadButton > button:hover {
            background-color: #00897b;
            font-color: white;
        }

        .stSpinner {
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------
# Placeholder implementations for missing modules
# Replace these with your actual implementations

def parse_medical_report(text):
    """
    Parses medical test results from raw text.
    Returns list of dicts like:
    [
        {'test_name': 'Hemoglobin', 'value': 13.2, 'unit': 'g/dL', 'normal_range': (12.0, 16.0)},
        ...
    ]
    """
    # Example dummy parsing logic for demonstration:
    results = []
    # Regex pattern to match simple test lines: TestName: value unit (normal_low - normal_high)
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
    Uses the AI model to generate an explanation for a single test result.
    Here we simulate with a placeholder string.
    """
    low, high = normal_range
    if value < low:
        status = "below normal range"
    elif value > high:
        status = "above normal range"
    else:
        status = "within normal limits"
    return f"The {test_name} result is {value} {unit}, which is {status}."

def preprocess_image(image_path):
    """
    Preprocesses the image to improve OCR accuracy.
    This is a placeholder that just opens the image.
    """
    image = Image.open(image_path)
    # Example: convert to grayscale (improve OCR)
    return image.convert("L")

def extract_text_from_image(image):
    """
    Extracts text from an image using OCR.
    Placeholder: returns dummy text.
    You can replace this with pytesseract or Google Vision OCR code.
    """
    # If you want to use pytesseract, uncomment below after installing pytesseract
    # import pytesseract
    # return pytesseract.image_to_string(image)
    return "Hemoglobin: 13.2 g/dL (12.0-16.0)\nWBC: 7000 /µL (4000-11000)"

# ---------------
# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

MAX_RETRIES = 3
RETRY_DELAY = 2

def analyze_medical_report(content, content_type):
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

    for attempt in range(MAX_RETRIES):
        try:
            if content_type == "image":
                response = model.generate_content([prompt, content])
            else:
                response = model.generate_content(f"{prompt}\n\n{content}")
            
            if not response.text.strip():
                st.warning("AI returned empty analysis.")
                return fallback_analysis(content, content_type)
            return response.text
        except exceptions.GoogleAPIError as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"API error. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)

def fallback_analysis(content, content_type):
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

def generate_pdf_report(text):
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
    st.subheader("🧠 Explanations by AI")
    for result in test_results:
        with st.expander(f"Explain {result['test_name']}"):
            explanation = generate_explanation_for_test(
                model,
                result['test_name'],
                result['value'],
                result['unit'],
                result['normal_range']
            )
            st.write(explanation)

# -------- Main App --------

st.title("💉 Medical Lab Report Analysis & Personalized Recommendations")

uploaded_file = st.file_uploader(
    "Upload your medical report (PDF, Image, or Text file)",
    type=["pdf", "png", "jpg", "jpeg", "txt"]
)

if uploaded_file:
    file_type = uploaded_file.type
    content_type = None
    report_text = ""

    if file_type == "application/pdf":
        content_type = "pdf"
        report_text = extract_text_from_pdf(uploaded_file)
        if not report_text.strip():
            st.warning("No text extracted from PDF. Try uploading an image or text file.")
    elif file_type.startswith("image/"):
        content_type = "image"
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        image = preprocess_image(tmp_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        report_text = extract_text_from_image(image)
        os.remove(tmp_path)
    elif file_type == "text/plain":
        content_type = "text"
        report_text = uploaded_file.read().decode("utf-8")
        st.text_area("Medical Report Text", value=report_text, height=250)
    else:
        st.error("Unsupported file type. Please upload PDF, Image, or Text file.")

    if report_text.strip():
        analysis_text = analyze_medical_report(report_text, content_type)
        st.subheader("📝 AI-Generated Analysis & Recommendations")
        st.write(analysis_text)

        # Display extracted table + explanations
        test_results = display_results_table(report_text)
        if test_results:
            display_explanations(test_results, model)

        # Offer PDF download
        pdf_buffer = generate_pdf_report(analysis_text)
        st.download_button(
            label="📥 Download AI Analysis as PDF",
            data=pdf_buffer,
            file_name="medical_report_analysis.pdf",
            mime="application/pdf"
        )

else:
    st.info("Please upload a medical report to get started.")


# ai_utils.py
import google.generativeai as genai
from google.api_core import exceptions
import time

MAX_RETRIES = 3
RETRY_DELAY = 2

def generate_explanation_for_test(model, test_name, value, unit, normal_range):
    prompt = (
        f"Explain in simple language what it means if the patient’s {test_name} is {value} {unit}, "
        f"given the normal range is {normal_range[0]} to {normal_range[1]}."
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            return response.text
        except exceptions.GoogleAPIError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "Explanation not available due to API error."

# parser.py
import re

def parse_medical_report(text):
    """
    Simple example parser to extract test data lines from report text.
    Assumes lines like: "Hemoglobin 9.5 g/dL (13-17)"
    Returns a list of dicts with keys: test_name, value, unit, normal_range
    """
    results = []

    # Example regex pattern (you may want to adjust based on your data)
    pattern = re.compile(r'(?P<test>[A-Za-z ]+)\s+(?P<value>\d+(\.\d+)?)\s*(?P<unit>[a-zA-Z/%]+)?\s*\(?(?P<range_low>\d+(\.\d+)?)-(?P<range_high>\d+(\.\d+)?)\)?')

    for line in text.split('\n'):
        match = pattern.search(line)
        if match:
            results.append({
                'test_name': match.group('test').strip(),
                'value': float(match.group('value')),
                'unit': match.group('unit') if match.group('unit') else '',
                'normal_range': (float(match.group('range_low')), float(match.group('range_high')))
            })
    return results



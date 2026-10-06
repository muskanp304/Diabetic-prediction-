"""
Diagnostic Report Parser for Diabetes Clinical Parameters
Supports: PDF, CSV, Excel, TXT, JSON medical reports.
Extracts: Glucose, BloodPressure, BMI, Insulin, Age, SkinThickness, Pregnancies, DiabetesPedigreeFunction.
"""

import io
import re
import json
import pandas as pd
import numpy as np

def extract_text_from_pdf(file_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_parameters_from_text(text):
    """
    Scans freeform medical text or OCR output for 8 Pima clinical parameters.
    Returns: dict of {parameter_name: {'value': float, 'raw_match': str}}
    """
    results = {}

    patterns = {
        'Glucose': [
            r'(?:plasma\s+fasting\s+glucose|fasting\s+plasma\s+glucose|fasting\s+blood\s+sugar|blood\s+glucose|plasma\s+glucose|fasting\s+sugar|glucose|sugar|fbs|rbs|ppbs)[\s:\-=_\n]*(\d+(?:\.\d+)?)',
            r'glucose[^\d\n]{0,30}\n\s*(\d+(?:\.\d+)?)',
            r'(?:glucose|sugar)[\s:]+(\d+(?:\.\d+)?)'
        ],
        'BloodPressure': [
            r'(?:diastolic\s+blood\s+pressure|diastolic\s+bp|diastolic)[\s:\-=_\n]*(\d{2,3})',
            r'(?:blood\s+pressure|b\.?p\.?)[\s:\-=_\n]*\d{2,3}\s*/\s*(\d{2,3})',  # 120/80 -> captures 80 (diastolic)
            r'(?:blood\s+pressure|bp)[\s:\-=_\n]*(\d{2,3})',
            r'blood\s+pressure[^\d\n]{0,30}\n\s*(\d{2,3})'
        ],
        'BMI': [
            r'(?:body\s+mass\s+index|b\.?m\.?i\.?)[\s:\-=_\n]*(\d{1,2}(?:\.\d+)?)',
            r'bmi[^\d\n]{0,30}\n\s*(\d{1,2}(?:\.\d+)?)',
            r'(?:bmi|body\s+mass\s+index)\s*\([^\)]*\)[\s:\-=_\n]*(\d{1,2}(?:\.\d+)?)'
        ],
        'Insulin': [
            r'(?:fasting\s+serum\s+insulin|serum\s+insulin|fasting\s+insulin|insulin)[\s:\-=_\n]*(\d+(?:\.\d+)?)',
            r'insulin[^\d\n]{0,30}\n\s*(\d+(?:\.\d+)?)'
        ],
        'SkinThickness': [
            r'(?:triceps\s+skinfold\s+thickness|triceps\s+skin\s+fold|skin\s+thickness|skinfold|triceps)[\s:\-=_\n]*(\d+(?:\.\d+)?)',
            r'skinfold[^\d\n]{0,30}\n\s*(\d+(?:\.\d+)?)'
        ],
        'Age': [
            r'(?:patient\s+age|age)[\s:\-=_\n]*(\d{1,3})\s*(?:years|yrs|y)?\b',
            r'age[^\d\n]{0,30}\n\s*(\d{1,3})'
        ],
        'Pregnancies': [
            r'(?:gravida|parity|pregnancies|number\s+of\s+pregnancies)[\s:\-=_\n]*(\d+)',
            r'pregnancies[^\d\n]{0,30}\n\s*(\d+)'
        ],
        'DiabetesPedigreeFunction': [
            r'(?:diabetes\s+pedigree\s+function|pedigree\s+function|pedigree|dpf)[\s:\-=_\n]*(\d+(?:\.\d+)?)',
            r'pedigree[^\d\n]{0,30}\n\s*(\d+(?:\.\d+)?)'
        ]
    }

    # Range validators for sanity check
    bounds = {
        'Glucose': (40, 400),
        'BloodPressure': (40, 160),
        'BMI': (10.0, 70.0),
        'Insulin': (1, 900),
        'SkinThickness': (5, 99),
        'Age': (1, 120),
        'Pregnancies': (0, 25),
        'DiabetesPedigreeFunction': (0.01, 2.8)
    }

    # Also detect patient name if available
    name_match = re.search(r'(?:patient\s*name|name\s*of\s*patient)[\s:\-=_\n]*([A-Za-z\s\.]{2,30}?)(?:\n|\r|date|age|gender|referred|sample|$)', text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r'patient[\s:\-=_\n]*([A-Za-z\s\.]{2,25}?)(?:\n|\r|date|age|gender|referred|$)', text, re.IGNORECASE)
    patient_name = name_match.group(1).strip() if name_match else "Patient"
    if len(patient_name) < 2 or any(k in patient_name.lower() for k in ['date', 'report', 'evaluation', 'investigation', 'center']):
        patient_name = "Patient"

    for param, pattern_list in patterns.items():
        found = False
        for p in pattern_list:
            matches = list(re.finditer(p, text, re.IGNORECASE))
            if matches:
                # Prioritize valid ranges
                for m in matches:
                    try:
                        val = float(m.group(1))
                        low, high = bounds[param]
                        if low <= val <= high:
                            results[param] = {
                                'value': int(val) if param in ['Pregnancies', 'Age', 'BloodPressure'] else round(val, 2),
                                'raw_match': m.group(0).strip().replace('\n', ' ')
                            }
                            found = True
                            break
                    except (ValueError, IndexError):
                        continue
            if found:
                break

    return patient_name, results

def parse_report_file(file_obj):
    """
    Accepts an uploaded file from Streamlit (file-like object or bytes),
    dispatches based on extension, and returns (extracted_dict, raw_text, patient_name).
    """
    filename = getattr(file_obj, 'name', 'uploaded_report.txt').lower()
    file_bytes = file_obj.read()
    
    extracted_params = {}
    raw_text = ""
    patient_name = "Patient"

    if filename.endswith('.pdf'):
        raw_text = extract_text_from_pdf(file_bytes)
        patient_name, extracted_params = extract_parameters_from_text(raw_text)

    elif filename.endswith('.csv'):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            raw_text = df.to_string()
            # If standard columns exist
            row = df.iloc[0]
            col_map = {
                'glucose': 'Glucose', 'fasting_glucose': 'Glucose', 'plasma_glucose': 'Glucose',
                'bloodpressure': 'BloodPressure', 'bp': 'BloodPressure', 'diastolic': 'BloodPressure',
                'bmi': 'BMI', 'body_mass_index': 'BMI',
                'insulin': 'Insulin', 'serum_insulin': 'Insulin',
                'skinthickness': 'SkinThickness', 'skin_thickness': 'SkinThickness',
                'age': 'Age', 'patient_age': 'Age',
                'pregnancies': 'Pregnancies', 'gravida': 'Pregnancies',
                'diabetespedigreefunction': 'DiabetesPedigreeFunction', 'pedigree': 'DiabetesPedigreeFunction', 'dpf': 'DiabetesPedigreeFunction'
            }
            # Search columns
            for col in df.columns:
                cleaned = re.sub(r'[^a-zA-Z]', '', col.lower())
                if cleaned in col_map:
                    target_key = col_map[cleaned]
                    val = row[col]
                    if pd.notna(val):
                        extracted_params[target_key] = {
                            'value': float(val),
                            'raw_match': f"Column '{col}' = {val}"
                        }
            if 'PatientName' in df.columns:
                patient_name = str(df['PatientName'].iloc[0])
        except Exception as e:
            raw_text = f"Error reading CSV: {e}"

    elif filename.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            raw_text = df.to_string()
            row = df.iloc[0]
            for col in df.columns:
                cleaned = re.sub(r'[^a-zA-Z]', '', col.lower())
                # match with parameters
                for param in ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction']:
                    if param.lower() in cleaned or cleaned in param.lower():
                        val = row[col]
                        if pd.notna(val):
                            extracted_params[param] = {'value': float(val), 'raw_match': f"Excel '{col}' = {val}"}
        except Exception as e:
            raw_text = f"Error reading Excel: {e}"

    elif filename.endswith('.json'):
        try:
            data = json.loads(file_bytes.decode('utf-8', errors='ignore'))
            raw_text = json.dumps(data, indent=2)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            for k, v in data.items():
                cleaned = re.sub(r'[^a-zA-Z]', '', k.lower())
                for param in ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction']:
                    if param.lower() in cleaned or cleaned in param.lower():
                        try:
                            extracted_params[param] = {'value': float(v), 'raw_match': f"JSON '{k}' = {v}"}
                        except:
                            pass
        except Exception as e:
            raw_text = f"Error reading JSON: {e}"

    else:
        # Fallback to plain text
        try:
            raw_text = file_bytes.decode('utf-8', errors='ignore')
            patient_name, extracted_params = extract_parameters_from_text(raw_text)
        except Exception as e:
            raw_text = f"Error reading text: {e}"

    return patient_name, extracted_params, raw_text

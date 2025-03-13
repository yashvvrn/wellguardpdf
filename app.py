from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from weasyprint import HTML, CSS
import os
import sys
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress WeasyPrint warnings
logging.getLogger('weasyprint').setLevel(logging.ERROR)

app = Flask(__name__)

# Get the absolute path to the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base directory: {BASE_DIR}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

def verify_data_files():
    """Verify the existence of required data files and print their status"""
    required_files = {
        'Data/Training.csv': False,
        'MasterData/Symptom_severity.csv': False,
        'MasterData/symptom_Description.csv': False,
        'MasterData/symptom_precaution.csv': False
    }
    
    for file_path in required_files:
        full_path = os.path.join(BASE_DIR, file_path)
        exists = os.path.isfile(full_path)
        required_files[file_path] = exists
        print(f"Checking {full_path}: {'EXISTS' if exists else 'MISSING'}")
    
    return all(required_files.values())

# Verify data files before proceeding
if not verify_data_files():
    print("ERROR: Required data files are missing!")
    sys.exit(1)

# --- Data Loading and Model Training ---
try:
    training_file = os.path.join(BASE_DIR, 'Data', 'Training.csv')
    print(f"Loading training data from: {training_file}")
    training = pd.read_csv(training_file)
    cols = training.columns[:-1]  # All symptom columns
    x = training[cols]
    y = training['prognosis']

    # Encode the prognosis labels
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    # Split data and train a Decision Tree classifier
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    # List of symptoms for the input vector and to populate the form
    symptoms_list = list(cols)
    print("Successfully loaded and processed training data")
except Exception as e:
    print(f"Error loading training data: {e}")
    raise

# --- Load Supplementary Data ---
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_severity_dict():
    global severityDictionary
    try:
        severity_file = os.path.join(BASE_DIR, 'MasterData', 'Symptom_severity.csv')
        print(f"Loading severity data from: {severity_file}")
        with open(severity_file, newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row:
                    symptom, severity = row[0], row[1]
                    severityDictionary[symptom] = int(severity)
        print(f"Successfully loaded {len(severityDictionary)} severity entries")
    except Exception as e:
        print(f"Error loading severity data: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        print(f"MasterData contents: {os.listdir('MasterData') if os.path.exists('MasterData') else 'MasterData NOT FOUND'}")
        raise

def load_description():
    global description_list
    try:
        with open(os.path.join(BASE_DIR, 'MasterData', 'symptom_Description.csv'), newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row:
                    symptom, desc = row[0], row[1]
                    description_list[symptom] = desc
    except Exception as e:
        print(f"Error loading description data: {e}")
        raise

def load_precaution_dict():
    global precautionDictionary
    try:
        with open(os.path.join(BASE_DIR, 'MasterData', 'symptom_precaution.csv'), newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row:
                    symptom = row[0]
                    precautions = row[1:5]  # Assumes 4 precaution entries per symptom
                    precautionDictionary[symptom] = precautions
    except Exception as e:
        print(f"Error loading precaution data: {e}")
        raise

# Load all data
load_severity_dict()
load_description()
load_precaution_dict()

# --- Helper Function for Severity Evaluation ---
def calc_condition(selected_symptoms, days):
    total = sum(severityDictionary.get(symptom, 0) for symptom in selected_symptoms)
    # Evaluate the condition based on severity and number of days
    if (total * days) / (len(selected_symptoms) + 1) > 13:
        return "Based on your presenting symptoms, a medical consultation is recommended."
    else:
        return "It may not be severe, but taking preventive measures is advisable."

# --- Single Route for Form and Result ---
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        name = request.form.get('name')
        selected_symptoms = request.form.getlist('symptoms')
        try:
            days = int(request.form.get('days'))
        except (TypeError, ValueError):
            days = 0  # Default if conversion fails

        # Build the input vector (one-hot encoding of selected symptoms)
        input_vector = np.zeros(len(symptoms_list))
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1

        # Predict disease using the trained Decision Tree
        prediction = clf.predict([input_vector])
        disease = le.inverse_transform(prediction)[0]

        # Retrieve disease information and precautions
        description = description_list.get(disease, "No description available.")
        precautions = precautionDictionary.get(disease, [])
        advice = calc_condition(selected_symptoms, days)

        result = {
            'name': name,
            'disease': disease,
            'description': description,
            'precautions': precautions,
            'advice': advice,
            'selected_symptoms': selected_symptoms,
            'days': days
        }
    return render_template('index.html', symptoms=symptoms_list, result=result)

# --- Route for PDF Report Download ---
@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        # Retrieve data from form submission
        name = request.form.get('name')
        selected_symptoms = request.form.get('selected_symptoms')
        days = request.form.get('days')
        disease = request.form.get('disease')
        description = request.form.get('description')
        advice = request.form.get('advice')
        precautions = request.form.getlist('precautions')

        logger.info(f"Generating PDF report for patient: {name}")
        logger.debug(f"Form data: {request.form}")

        # Convert selected_symptoms to a list (if comma separated)
        symptoms_from_form = selected_symptoms.split(", ") if selected_symptoms else []

        result = {
            "name": name,
            "selected_symptoms": symptoms_from_form,
            "days": days,
            "disease": disease,
            "description": description,
            "precautions": precautions,
            "advice": advice
        }
        
        # Render the styled HTML report
        rendered = render_template('report.html', result=result)
        
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            # Define CSS for PDF styling
            css = CSS(string='''
                @page { 
                    size: A4; 
                    margin: 1cm;
                    @top-center {
                        content: "WellGuard Medical Report";
                        font-size: 9pt;
                    }
                    @bottom-center {
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 9pt;
                    }
                }
                body { font-family: sans-serif; }
            ''')
            
            # Generate PDF with custom CSS
            HTML(string=rendered).write_pdf(
                target=tmp.name,
                stylesheets=[css]
            )
            
            logger.info("PDF generation completed")
            
            # Read the generated PDF
            with open(tmp.name, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
            
            # Clean up the temporary file
            os.unlink(tmp.name)
        
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={name}_medical_report.pdf'
        return response

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        return f"Error generating PDF report: {str(e)}", 500

@app.route('/health')
def health_check():
    return 'OK', 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)

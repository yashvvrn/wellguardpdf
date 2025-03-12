from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from weasyprint import HTML
import os

app = Flask(__name__)

# Update paths to be relative to the api directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Loading and Model Training ---
training = pd.read_csv(os.path.join(BASE_DIR, 'Data/Training.csv'))
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

# --- Load Supplementary Data ---
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_severity_dict():
    global severityDictionary
    with open(os.path.join(BASE_DIR, 'MasterData/symptom_severity.csv'), newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row:
                symptom, severity = row[0], row[1]
                severityDictionary[symptom] = int(severity)

def load_description():
    global description_list
    with open(os.path.join(BASE_DIR, 'MasterData/symptom_Description.csv'), newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row:
                symptom, desc = row[0], row[1]
                description_list[symptom] = desc

def load_precaution_dict():
    global precautionDictionary
    with open(os.path.join(BASE_DIR, 'MasterData/symptom_precaution.csv'), newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row:
                symptom = row[0]
                precautions = row[1:5]  # Assumes 4 precaution entries per symptom
                precautionDictionary[symptom] = precautions

load_severity_dict()
load_description()
load_precaution_dict()

# --- Helper Function for Severity Evaluation ---
def calc_condition(selected_symptoms, days):
    total = sum(severityDictionary.get(symptom, 0) for symptom in selected_symptoms)
    if (total * days) / (len(selected_symptoms) + 1) > 13:
        return "Based on your presenting symptoms, a medical consultation is recommended."
    else:
        return "It may not be severe, but taking preventive measures is advisable."

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        name = request.form.get('name')
        selected_symptoms = request.form.getlist('symptoms')
        try:
            days = int(request.form.get('days'))
        except (TypeError, ValueError):
            days = 0

        input_vector = np.zeros(len(symptoms_list))
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1

        prediction = clf.predict([input_vector])
        disease = le.inverse_transform(prediction)[0]

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

@app.route('/download_report', methods=['POST'])
def download_report():
    name = request.form.get('name')
    selected_symptoms = request.form.get('selected_symptoms')
    days = request.form.get('days')
    disease = request.form.get('disease')
    description = request.form.get('description')
    advice = request.form.get('advice')
    precautions = request.form.getlist('precautions')

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
    
    rendered = render_template('report.html', result=result)
    pdf = HTML(string=rendered).write_pdf()
    
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response

# For local development
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True) 
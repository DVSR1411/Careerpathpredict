from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open("career_pkl.pkl", "rb"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'database': request.form['database'],
        'computer_architecture': request.form['computer_architecture'],
        'distributed_computing': request.form['distributed_computing'],
        'cyber_security': request.form['cyber_security'],
        'networking': request.form['networking'],
        'software_development': request.form['software_development'],
        'programming_skills': request.form['programming_skills'],
        'project_management': request.form['project_management'],
        'computer_forensics': request.form['computer_forensics'],
        'technical_communication': request.form['technical_communication'],
        'ai_ml': request.form['ai_ml'],
        'software_engineering': request.form['software_engineering'],
        'business_analysis': request.form['business_analysis'],
        'communication_skills': request.form['communication_skills'],
        'data_science': request.form['data_science'],
        'troubleshooting_skills': request.form['troubleshooting_skills'],
        'graphics_designing': request.form['graphics_designing']
    }
    mapping = {
        'Not Interested': 4,
        'Poor': 5,
        'Beginner': 1,
        'Average': 0,
        'Intermediate': 3,
        'Excellent': 2,
        'Professional': 6
    }
    for key, value in data.items():
        data[key] = mapping[value]
    data_array = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(data_array)
    return render_template('index.html', prediction=prediction[0])
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

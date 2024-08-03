from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('heart_disease_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            data = request.form
            age = int(data['age'])
            sex = int(data['sex'])
            cp = int(data['cp'])
            trestbps = int(data['trestbps'])
            chol = int(data['chol'])
            fbs = int(data['fbs'])
            restecg = int(data['restecg'])
            thalach = int(data['thalach'])
            exang = int(data['exang'])
            oldpeak = float(data['oldpeak'])
            slope = int(data['slope'])
            ca = int(data['ca'])
            thal = int(data['thal'])

            # Create input data array with 13 features
            sample_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            # Standardize the input data
            sample_data = scaler.transform(sample_data)

            # Make the prediction
            prediction = model.predict(sample_data)
            result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

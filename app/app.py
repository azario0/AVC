from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and preprocessing objects
model = joblib.load('random_forest_stroke_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_ever_married = joblib.load('le_ever_married.pkl')
le_work_type = joblib.load('le_work_type.pkl')
le_residence_type = joblib.load('le_residence_type.pkl')
le_smoking_status = joblib.load('le_smoking_status.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user inputs
        gender = le_gender.transform([request.form['gender']])[0]
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = le_ever_married.transform([request.form['ever_married']])[0]
        work_type = le_work_type.transform([request.form['work_type']])[0]
        residence_type = le_residence_type.transform([request.form['residence_type']])[0]
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = le_smoking_status.transform([request.form['smoking_status']])[0]

        # Prepare the input for prediction
        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, 
                                work_type, residence_type, avg_glucose_level, bmi, smoking_status]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        result = "High risk of stroke" if prediction == 1 else "Low risk of stroke"

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
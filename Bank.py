from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('lgbm_churn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    features = [
        float(request.form['Customer_Age']),
        float(request.form['Avg_Utilization_Ratio']),
        float(request.form['Total_Ct_Chng_Q4_Q1']),
        float(request.form['Total_Trans_Ct']),
        float(request.form['Total_Trans_Amt']),
        float(request.form['Total_Revolving_Bal']),
        float(request.form['Months_Inactive_12_mon']),
        float(request.form['Contacts_Count_12_mon']),
        float(request.form['Total_Relationship_Count']),
        # ...add all necessary features
    ]

    prediction = model.predict([features])[0]
    result = 'This Customer is likely to Churn' if prediction == 1 else 'This Customer is not likely to Churn'
    
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)

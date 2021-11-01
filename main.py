from logging import debug
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#load model
model = joblib.load('hiring_model.pkl')


@app.route('/')

def page():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    exp = request.form.get('experience')
    sc = request.form.get('test_score')
    int_sc = request.form.get('Interview_score')
    
    prediction = model.predict([[exp, sc, int_sc]])

    output = round(prediction[0],2)

    return render_template('base.html', prediction_text = f"Employee salary will be $ {output}")

if __name__ == '__main__':
    app.run(debug=True)
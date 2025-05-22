from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

## importing the model (ridge and standard scaler pickle files)
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

@app.route('/')

def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None  # define result at the beginning

    if request.method == 'POST':
        # Get the input values from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        New_Scaled_Data = standard_scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(New_Scaled_Data)[0]  # get just the value

    return render_template('home.html', results=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
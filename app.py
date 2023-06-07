import os
from flask import Flask, render_template, request
import pickle

# Get the absolute path to the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get the input data from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])
    feature4 = float(request.form['feature4'])

    # Scale the input features
    input_features = scaler.transform([[feature1, feature2, feature3, feature4]])

    # Make the prediction
    prediction = model.predict(input_features)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)


from flask import Flask, render_template, request, redirect, url_for
import json
import pickle

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.p'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load the results.json file
def load_results():
    with open('results.json', 'r') as f:
        return json.load(f)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        # Perform necessary file handling (if needed)
        return "File uploaded successfully!"
    return render_template('upload.html')

# Results route
@app.route('/results')
def results():
    results = load_results()
    return render_template('results.html', results=results)

# Translationa route
@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        input_text = request.form['input_text']
        # Perform inference using the model
        prediction = model.predict([input_text])  # Example usage
        return render_template('translate.html', input_text=input_text, prediction=prediction[0])
    return render_template('translate.html')

if __name__ == '__main__':
    app.run(debug=True)

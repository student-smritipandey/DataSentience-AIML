from flask import Flask, request, url_for, redirect, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model if it exists, otherwise create it
model_path = 'model.pkl'
try:
    if not os.path.exists(model_path):
        # Import and run the model training script
        import mental_health
        model = mental_health.model
    else:
        model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a simple fallback model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    # Train with dummy data
    X_dummy = np.array([[25, 0, 0], [30, 1, 1], [35, 0, 0], [28, 1, 0]])
    y_dummy = np.array([0, 1, 0, 1])
    model.fit(X_dummy, y_dummy)

@app.route('/')
def hello_world():
    # Read the HTML template directly
    with open('index.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    # Read the HTML template
    with open('index.html', 'r') as f:
        html_content = f.read()

    if float(output) > 0.5:
        result = f'You need a treatment.\nProbability of mental illness is {output}'
    else:
        result = f'You do not need treatment.\n Probability of mental illness is {output}'

    # Replace the placeholder with the result
    html_content = html_content.replace('{{pred}}', result)
    return html_content

if __name__ == '__main__':
    app.run(debug=True)

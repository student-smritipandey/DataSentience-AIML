# Mental Health Predictor - Web Interface

A Flask-based web application for predicting mental health conditions using machine learning.

## Features

- Interactive web form for mental health assessment
- Real-time prediction using stacking classifier model
- Probability-based treatment recommendations
- Responsive design with Materialize CSS

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python app.py
   ```

3. **Access the Web Interface:**
   Open your browser and go to `http://localhost:5000`

## Usage

1. Enter your **Age** in years
2. Select your **Gender**:
   - 0 for Male
   - 1 for Female  
   - 2 for Transgender
3. Indicate **Family History**:
   - 0 for No
   - 1 for Yes
4. Click "Predict Probability" to get your mental health assessment

## Model Information

- **Algorithm**: Stacking Classifier (KNN + Random Forest + Naive Bayes)
- **Features**: Age, Gender, Family History
- **Threshold**: 0.5 probability for treatment recommendation

## Files Structure

- `app.py` - Flask web application
- `mental_health.py` - Model training and preprocessing
- `index.html` - Web interface template
- `style.css` - Custom styling
- `mental_health.csv` - Training dataset
- `requirements.txt` - Python dependencies 
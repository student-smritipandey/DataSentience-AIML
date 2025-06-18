# Mental-Health-Prediction-using-Machine-Learning-Algorithms
Prediction of Mental Health using various Machine Learning Algorithms and made a Web page which will predict the probability of Mental illness based on inputs provided by user.

# It can be easily tested now by the Web Based Interface
Mental Health Predictor - Web Interface
A Flask-based web application for predicting mental health conditions using machine learning.

Features
Interactive web form for mental health assessment
Real-time prediction using stacking classifier model
Probability-based treatment recommendations
Responsive design with Materialize CSS

Initial view after Launch:
![Screenshot 2025-06-18 101826](https://github.com/user-attachments/assets/183074bf-8f19-40c1-b7b5-7f7b6e7dafaa)

Test Results: 
![Screenshot 2025-06-18 101917](https://github.com/user-attachments/assets/29792822-6373-49ac-b93f-b19b6639d4ec)


Setup Instructions
Install Dependencies:
pip install -r requirements.txt

Run the Application:
python app.py
Access the Web Interface: Open your browser and go to http://localhost:5000

Usage
Enter your Age in years
Select your Gender:
0 for Male
1 for Female
2 for Transgender
Indicate Family History:
0 for No
1 for Yes
Click "Predict Probability" to get your mental health assessment

Model Information
Algorithm: Stacking Classifier (KNN + Random Forest + Naive Bayes)
Features: Age, Gender, Family History
Threshold: 0.5 probability for treatment recommendation

Files Structure
app.py - Flask web application
mental_health.py - Model training and preprocessing
index.html - Web interface template
style.css - Custom styling
mental_health.csv - Training dataset
requirements.txt - Python dependencies

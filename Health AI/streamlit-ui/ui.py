import streamlit as st
import pandas as pd
import numpy as np
import joblib

symptom_list = ['shortness of breath', 'dizziness', 'asthenia', 'fall', 'syncope',
       'vertigo', 'sweat', 'sweating increased', 'palpitation', 'nausea',
       'angina pectoris', 'pressure chest', 'polyuria', 'polydypsia',
       'pain chest', 'orthopnea', 'rale', 'unresponsiveness',
       'mental status changes', 'vomiting', 'labored breathing',
       'feeling suicidal', 'suicidal', 'hallucinations auditory',
       'feeling hopeless', 'weepiness', 'sleeplessness',
       'motor retardation', 'irritable mood', 'blackout',
       'mood depressed', 'hallucinations visual', 'worry', 'agitation',
       'tremor', 'intoxication', 'verbal auditory hallucinations',
       'energy increased', 'difficulty', 'nightmare',
       'unable to concentrate', 'homelessness', 'hypokinesia',
       'dyspnea on exertion', 'chest tightness', 'cough', 'fever',
       'decreased translucency', 'productive cough', 'pleuritic pain',
       'yellow sputum', 'breath sounds decreased', 'chill', 'rhonchus','green sputum', 'non-productive cough', 'wheezing', 'haemoptysis',
       'distress respiratory', 'tachypnea', 'malaise', 'night sweat',
       'jugular venous distention', 'dyspnea', 'dysarthria',
       'speech slurred', 'facial paresis', 'hemiplegia', 'seizure',
       'numbness', 'symptom aggravating factors', 'st segment elevation',
       'st segment depression', 't wave inverted', 'presence of q wave',
       'chest discomfort', 'bradycardia', 'pain', 'nonsmoker', 'erythema',
       'hepatosplenomegaly', 'pruritus', 'diarrhea', 'abscess bacterial',
       'swelling', 'apyrexial', 'dysuria', 'hematuria',
       'renal angle tenderness', 'lethargy', 'hyponatremia',
       'hemodynamically stable', 'difficulty passing urine',
       'consciousness clear', 'guaiac positive', 'monoclonal',
       'ecchymosis', 'tumor cell invasion', 'haemorrhage', 'pallor',
       'fatigue', 'heme positive', 'pain back', 'orthostasis',
       'arthralgia', 'transaminitis', 'sputum purulent', 'hypoxemia',
       'hypercapnia', 'patient non compliance', 'unconscious state',
       'bedridden', 'abdominal tenderness', 'unsteady gait',
       'hyperkalemia', 'urgency of\xa0micturition', 'ascites',
       'hypotension', 'enuresis', 'asterixis', 'muscle twitch', 'sleepy',
       'headache', 'lightheadedness', 'food intolerance',
       'numbness of hand', 'general discomfort', 'drowsiness',
       'stiffness', 'prostatism', 'weight gain', 'tired',
       'mass of body structure', 'has religious belief', 'nervousness',
       'formication', 'hot flush', 'lesion', 'cushingoid facies',
       'cushingoid\xa0habitus', 'emphysematous change',
       'decreased body weight', 'hoarseness', 'thicken',
       'spontaneous rupture of membranes', 'muscle hypotonia',
       'hypotonic', 'redness', 'hypesthesia', 'hyperacusis','scratch marks', 'sore to touch', 'burning sensation',
       'satiety early', 'throbbing sensation quality',
       'sensory discomfort', 'constipation', 'pain abdominal',
       'heartburn', 'breech presentation', 'cyanosis',
       'pain in lower limb', 'cardiomegaly', 'clonus', 'unwell',
       'anorexia', 'history of - blackout', 'anosmia',
       'metastatic lesion', 'hemianopsia homonymous',
       'hematocrit decreased', 'neck stiffness', 'cicatrisation',
       'hypometabolism', 'aura', 'myoclonus', 'gurgle',
       'wheelchair bound', 'left\xa0atrial\xa0hypertrophy', 'oliguria',
       'catatonia', 'unhappy', 'paresthesia', 'gravida 0', 'lung nodule',
       'distended abdomen', 'ache', 'macerated skin', 'heavy feeling',
       'rest pain', 'sinus rhythm', 'withdraw', 'behavior hyperactive',
       'terrify', 'photopsia', 'giddy mood', 'disturbed family',
       'hypersomnia', 'hyperhidrosis disorder', 'mydriasis',
       'extrapyramidal sign', 'loose associations', 'exhaustion', 'snore',
       'r wave feature', 'overweight', 'systolic murmur', 'asymptomatic',
       'splenomegaly', 'bleeding of vagina', 'macule', 'photophobia',
       'painful swallowing', 'cachexia', 'hypocalcemia result',
       'hypothermia, natural', 'atypia', 'general unsteadiness',
       'throat sore', 'snuffle', 'hacking cough', 'stridor', 'paresis',
       'aphagia', 'focal seizures', 'abnormal sensation', 'stupor',
       'fremitus', "Stahli's line", 'stinging sensation', 'paralyse',
       'hirsutism', 'sniffle', 'bradykinesia', 'out of breath',
       'urge incontinence', 'vision blurred', 'room spinning',
       'rambling speech', 'clumsiness', 'decreased stool caliber',
       'hematochezia', 'egophony', 'scar tissue', 'neologism',
       'decompensation', 'stool color yellow','rigor - temperature-associated observation', 'paraparesis',
       'moody', 'fear of falling', 'spasm', 'hyperventilation',
       'excruciating pain', 'gag', 'posturing', 'pulse absent',
       'dysesthesia', 'polymyalgia', 'passed stones',
       'qt interval prolonged', 'ataxia', "Heberden's node",
       'hepatomegaly', 'sciatica', 'frothy sputum', 'mass in breast',
       'retropulsion', 'estrogen use', 'hypersomnolence', 'underweight',
       'dullness', 'red blotches', 'colic abdominal', 'hypokalemia',
       'hunger', 'prostate tender', 'pain foot', 'urinary hesitation',
       'disequilibrium', 'flushing', 'indifferent mood', 'urinoma',
       'hypoalbuminemia', 'pustule', 'slowing of urinary stream',
       'extreme exhaustion', 'no status change', 'breakthrough pain',
       'pansystolic murmur', 'systolic ejection murmur', 'stuffy nose',
       'barking cough', 'rapid shallow breathing', 'noisy respiration',
       'nasal discharge present', 'frail', 'cystic lesion',
       'projectile vomiting', 'heavy legs', 'titubation',
       'dysdiadochokinesia', 'achalasia', 'side pain', 'monocytosis',
       'posterior\xa0rhinorrhea', 'incoherent', 'lameness',
       'claudication', 'clammy skin', 'mediastinal shift',
       'nausea and vomiting', 'awakening early', 'tenesmus', 'fecaluria',
       'pneumatouria', 'todd paralysis', 'alcoholic withdrawal symptoms',
       'myalgia', 'dyspareunia', 'poor dentition', 'floppy',
       'inappropriate affect', 'poor feeding', 'moan', 'welt', 'tinnitus',
       'hydropneumothorax', 'superimposition', 'feeling strange',
       'uncoordination', 'absences finding', 'tonic seizures',
       'debilitation', 'impaired cognition', 'drool', 'pin-point pupils',
       'tremor resting', 'groggy', 'adverse reaction', 'adverse effect',
       'abdominal bloating', 'fatigability', 'para 2', 'abortion',
       'intermenstrual heavy bleeding', 'previous pregnancies 2',
       'primigravida', 'abnormally hard consistency', 'proteinemia',
       'pain neck', 'dizzy spells', 'shooting pain', 'hyperemesis',
       'milky', 'regurgitates after swallowing', 'lip smacking',
       'phonophobia', 'rolling of eyes', 'ambidexterity',
       'pulsus\xa0paradoxus', 'gravida 10', 'bruit',
       'breath-holding spell', 'scleral\xa0icterus', 'retch', 'blanch',
       'elation', 'verbally abusive behavior', 'transsexual',
       'behavior showing increased motor activity',
       'coordination abnormal', 'choke', 'bowel sounds decreased',
       'no known drug allergies', 'low back pain', 'charleyhorse',
       'sedentary', 'feels hot/feverish', 'flare',
       'pericardial friction rub', 'hoard', 'panic',
       'cardiovascular finding', 'cardiovascular event',
       'soft tissue swelling', 'rhd positive', 'para 1', 'nasal flaring',
       'sneeze', 'hypertonicity', "Murphy's sign", 'flatulence',
       'gasping for breath', 'feces in rectum', 'prodrome',
       'hypoproteinemia', 'alcohol binge episode', 'abdomen acute',
       'air fluid level', 'catching breath', 'large-for-dates fetus',
       'immobile', 'homicidal thoughts']

# Load trained models (update paths as per your folder structure)
@st.cache_resource
def load_models():
    diabetes_model = joblib.load("diabetes_rf_model.pkl")
    heart_model = joblib.load("heart_disease_model.pkl")
    symptom_model = joblib.load("disease_predictor.pkl")
    return diabetes_model, heart_model, symptom_model

diabetes_model, heart_model, symptom_model = load_models()

# Sidebar Navigation
st.sidebar.title("Disease Prediction Suite")
app_mode = st.sidebar.radio("Choose Predictor", ["Diabetes Predictor", "Heart Disease Predictor", "Symptom-based Disease Predictor"])

if app_mode == "Diabetes Predictor":
    st.title("Diabetes Prediction")
    
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "ever", "not current", "No Info"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    HbA1c_level = st.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

    if st.button("Predict Diabetes"):
        # Encode categorical variables appropriately (should match training preprocessing)
        gender_encoded = 1 if gender == "Male" else 0
        smoking_map = {"never": 4, "former": 1, "current": 0, "ever": 2, "not current": 3, "No Info": 5}
        smoking_encoded = smoking_map[smoking_history]

        features = np.array([[gender_encoded, age, hypertension, heart_disease,
                              smoking_encoded, bmi, HbA1c_level, blood_glucose_level]])

        prediction = diabetes_model.predict(features)[0]
        st.success("Diabetic" if prediction == 1 else "Not Diabetic")

elif app_mode == "Heart Disease Predictor":
    st.title("Heart Disease Prediction")

    age = st.slider("Age", 20, 100)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200)
    chol = st.slider("Serum Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    if st.button("Predict Heart Disease"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(features)[0]
        st.success("Heart Disease Detected" if prediction == 1 else "No Heart Disease")

elif app_mode == "Symptom-based Disease Predictor":
    st.title("Disease Prediction from Symptoms")

    # Example: select multiple symptoms
    selected_symptoms = st.multiselect("Select symptoms:", symptom_list)


    if st.button("Predict Disease"):
        # Create input vector (example assumes binary vector)
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        prediction = symptom_model.predict([input_vector])[0]
        st.success(f"Predicted Disease: {prediction}")

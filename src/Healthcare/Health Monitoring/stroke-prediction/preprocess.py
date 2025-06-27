import pandas as pd

def preprocess_data(df):
    df = df.drop(columns=['id'], errors='ignore')

    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['gender'] = df['gender'].replace('Other', 'Unknown')

    df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

    X = df.drop(columns=['stroke'], errors='ignore')
    y = df['stroke'] if 'stroke' in df.columns else None

    return X, y

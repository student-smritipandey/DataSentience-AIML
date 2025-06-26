# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import preprocess_data

def train_model(csv_path=r"data/loan_train.csv"):
    df = pd.read_csv(csv_path)

    # Drop Loan_ID (if present)
    if 'Loan_ID' in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    # Map target column
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    else:
        raise ValueError("❌ 'Loan_Status' column not found in dataset.")

    # Preprocess (for training=True)
    df = preprocess_data(df, for_training=True)

    # Split features and label
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/model.pkl")
    print("✅ Model trained and saved to models/model.pkl")

if __name__ == "__main__":
    train_model()

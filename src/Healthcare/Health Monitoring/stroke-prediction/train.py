import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from preprocess import preprocess_data

# Load and preprocess the dataset
df = pd.read_csv("data/stroke_data.csv")
X, y = preprocess_data(df)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Apply SMOTE for class balancing
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
print("After SMOTE:", dict(pd.Series(y_train_resampled).value_counts()))

# Train model with balanced data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and training columns
os.makedirs("models", exist_ok=True)
joblib.dump((clf, X_train.columns.tolist()), "models/model.pkl")

print("\n✅ Model trained and saved to models/model.pkl")

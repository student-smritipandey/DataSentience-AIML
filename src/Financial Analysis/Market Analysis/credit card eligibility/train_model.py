import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load cleaned data
df = pd.read_csv("data/dataset.csv")

# Preprocess
X = pd.get_dummies(df.drop("Target", axis=1))
y = df["Target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ✅ SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ✅ Train
model = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42)
model.fit(X_train_smote, y_train_smote)
# Sanity check - test prediction on known eligible row
eligible_sample = X_train_smote[y_train_smote == 1].iloc[0]
print("✅ Should return 1 ->", model.predict([eligible_sample])[0])

# ✅ Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ Save trained model
pickle.dump(model, open("model/credit_model.pkl", "wb"))

# ✅ Sanity check: test model on a real eligible sample
sample = X_train_smote[y_train_smote == 1].iloc[0]
print("Sanity check prediction:", model.predict([sample])[0])

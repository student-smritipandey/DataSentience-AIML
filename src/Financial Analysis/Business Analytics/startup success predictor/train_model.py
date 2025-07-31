import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_clean_data

def train_and_save_model(filepath, model_path='models/rf_model.pkl'):
    X, y, _ = load_and_clean_data(filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_and_save_model("data/startup data.csv")

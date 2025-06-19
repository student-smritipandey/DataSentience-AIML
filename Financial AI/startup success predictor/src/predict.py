import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import load_and_clean_data

def predict_new_data(new_data_path, model_path='models/rf_model.pkl'):
    model = joblib.load(model_path)
    X, y, _ = load_and_clean_data(new_data_path)
    predictions = model.predict(X)

    result_df = pd.DataFrame({"Prediction": predictions})

    if y is not None:
        accuracy = accuracy_score(y, predictions)
        print("\nEvaluation Metrics:")
        print("Accuracy:", accuracy)
        print(classification_report(y, predictions))

        # Add ground truth and correctness to the output
        result_df["Actual"] = y.values
        result_df["Correct"] = result_df["Prediction"] == result_df["Actual"]

    print("\nPrediction Results (first 10 rows):")
    print(result_df.head(10).to_string(index=False))

    return predictions

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_csv>")
    else:
        result = predict_new_data(sys.argv[1])
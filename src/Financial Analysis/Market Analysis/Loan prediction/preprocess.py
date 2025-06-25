import pandas as pd

TRAIN_COLS = "models/train_columns.txt"

def preprocess_data(df, for_training=True):
    df = df.copy()

    # Drop Loan_Status during prediction
    if not for_training and 'Loan_Status' in df.columns:
        df = df.drop(columns=['Loan_Status'])

    # Fill missing values
    fill_strategies = {
        'Gender': 'mode',
        'Married': 'mode',
        'Dependents': 'mode',
        'Self_Employed': 'mode',
        'LoanAmount': 'median',
        'Loan_Amount_Term': 'mode',
        'Credit_History': 'mode'
    }

    for col, strategy in fill_strategies.items():
        if col in df.columns:
            if strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # One-hot encode categorical features
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
    df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=True)

    if for_training:
        # Save only feature columns (exclude target)
        columns_to_save = [col for col in df.columns if col != "Loan_Status"]
        with open(TRAIN_COLS, "w") as f:
            f.write(",".join(columns_to_save))
        return df

    # Load training-time feature columns
    with open(TRAIN_COLS, "r") as f:
        train_columns = f.read().split(",")

    # Add any missing columns from training
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[train_columns]

    return df

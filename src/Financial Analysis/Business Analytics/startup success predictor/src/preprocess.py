import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    drop_cols = ['Unnamed: 0', 'id', 'object_id', 'name', 'Unnamed: 6', 'city', 'zip_code',
                 'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at', 'state_code', 'state_code.1']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df['status'] = df['status'].apply(lambda x: 1 if x in ['operating', 'acquired'] else 0)
    df.fillna(0, inplace=True)

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['status'])
    y = df['status']

    return X, y, label_encoders
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=["type", "posts"], inplace=True)
    return df["posts"].values, df["type"].values

def preprocess(texts, vectorizer=None, fit=False):
    if fit:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer

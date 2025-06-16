import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import (
    confusion_matrix, roc_curve,roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Cancer Prediction System",
    page_icon="🔬",
    layout="wide"
)

with st.sidebar:
    st.title("Cancer Prediction System🔬")
    st.subheader("About")
    st.write("""
    Welcome to the AI-Powered Cancer Prediction System!✨

    This advanced tool, powered by machine learning and Streamlit, assists in predicting whether a tumor is benign or malignant based on various medical features.

    **Features:**
    - **Data Analysis**: Visualize key statistics and distributions.
    - **Cancer Prediction**: Input patient data to receive a prediction.
    - **Model Performance**: Evaluate accuracy, ROC curve, and confusion matrix.
    - **Feature Importance**: Identify the most influential factors.

    Enhance early detection and diagnosis with this smart system.
    """)
    def print_praise():
        praise_quotes = """
        Prerita Saini✨
        """
        title = "**Created By -**\n\n"
        return title + praise_quotes

    st.write(print_praise())

# Load dataset
file_path = "cancer.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
st.title("🔬 Advanced Cancer Prediction System")
st.write("### AI-Powered Diagnostic Support Tool")

df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')  # Remove unnecessary columns
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convert target to numeric

# Splitting dataset
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
log_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save models
with open("models.pkl", "wb") as f:
    pickle.dump((scaler, log_model, rf_model), f)

# Tabs for different sections
tabs = st.tabs([
    "📝 Predict Cancer", 
    "📊 Model Performance", 
    "🌳 Decision Tree", 
    "🔍 Feature Analysis & Insights"
])

# ✅ **Tab 1: User Input & Prediction**
with tabs[0]:
    st.write("Enter patient details below to predict cancer risk:")
    
    # User inputs for features
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    
    user_df = pd.DataFrame([user_input])
    user_df_scaled = scaler.transform(user_df)
    
    if st.button("🔍 Predict"):
        prediction = rf_model.predict(user_df_scaled)[0]
        result = "🛑 Malignant (Cancer Detected)" if prediction == 1 else "✅ Benign (No Cancer)"
        st.success(f"**Prediction: {result}**")

# ✅ **Tab 2: Model Performance**
with tabs[1]:

    # Statistical Summary
    st.subheader("Statistical Summary")
    st.write(df.describe())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, rf_model.predict(X_test))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"], ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("ROC Curve")
        y_probs = rf_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        st.pyplot(fig)

    st.subheader("Feature Importance")
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    feature_importances.nlargest(10).plot(kind="barh", ax=ax)
    st.pyplot(fig)

    # Tumor Texture Distribution
    st.subheader("Benign Tumor Distribution")
    fig, ax = plt.subplots()  # Adjusted figure size
    sns.histplot(df[df['diagnosis'] == 0]['texture_mean'], bins=30, color='blue', ax=ax)
    ax.set_title("Benign texture_mean Distribution")
    st.pyplot(fig)
    st.subheader("Malignant Tumor Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[df['diagnosis'] == 1]['texture_mean'], bins=30, color='red', ax=ax)
    ax.set_title("Malignant texture_mean Distribution")
    st.pyplot(fig)

# ✅ **Tab 3: Decision Tree Visualization**
with tabs[2]:
    
    st.subheader("🌳 Decision Tree Visualization")
    max_depth = st.slider("Select Tree Depth", 1, 5, 3)
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(rf_model.estimators_[0], feature_names=X.columns, filled=True, max_depth=max_depth)
    st.pyplot(fig)
    
    st.subheader("Decision Tree Summary")
    st.write(f"Tree Depth: {rf_model.estimators_[0].get_depth()}")
    st.write(f"Number of Nodes: {rf_model.estimators_[0].get_n_leaves()}")

    
    st.subheader("Top 10 Important Features")
    st.write(feature_importances.nlargest(10))
    

# ✅ **Tab 4: Feature Analysis & Insights**
with tabs[3]:
    
    st.subheader("📍Feature Relationship Visualization")
    feature_x = st.selectbox("Select Feature for X-axis", X.columns)
    feature_y = st.selectbox("Select Feature for Y-axis", X.columns)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df["diagnosis"], palette=["green", "red"], ax=ax)
    ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y}")
    st.pyplot(fig)
    
    st.subheader("🔥 Correlation Heatmap")
    threshold = st.slider("Select Correlation Threshold", 0.1, 1.0, 0.5)
    corr_matrix = df.corr()
    filtered_corr = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]
    top_corr_features = filtered_corr.abs().sum().sort_values(ascending=False).index[:10]
    filtered_corr = filtered_corr.loc[top_corr_features, top_corr_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Filtered Correlation Heatmap")
    st.pyplot(fig)

    st.write("### Insights:")
    st.info("🔹 **Scatter Plot** shows relationships between features.")
    st.success("🔹 **Heatmap** highlights strongly correlated features.")

    st.subheader("📊 Feature Distribution Analysis")
    feature = st.selectbox("Select a Feature", X.columns)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[feature], bins=30, kde=True, color="blue", ax=ax)
    ax.set_title(f"Histogram of {feature}")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(df[feature], fill=True, color="green", ax=ax)
    ax.set_title(f"KDE Plot of {feature}")
    st.pyplot(fig)

    st.write("### Feature Distribution Insights:")
    st.info("🔹 **Histogram** provides an overview of how a feature's values are distributed.")
    st.success("🔹 **KDE Plot** helps visualize the probability density of feature values.")




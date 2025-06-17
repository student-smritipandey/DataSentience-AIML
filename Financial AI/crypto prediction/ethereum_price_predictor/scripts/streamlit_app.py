import streamlit as st
from predict import predict_next_price

def main():
    # Page config
    st.set_page_config(page_title="Ethereum Price Predictor", page_icon="🪙", layout="centered")

    # Custom styling
    st.markdown("""
        <style>
        body {
            background: linear-gradient(to right, #e0eafc, #cfdef3);
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
            color: #1f2937;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton button {
            background-color: #6366f1;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stButton button:hover {
            background-color: #4f46e5;
        }
        .white-text {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("## 🪙 Ethereum Price Predictor")
    
    # White subtitle text
    st.markdown('<p class="white-text">Predict the <strong>next closing price</strong> of Ethereum (ETH) using historical market data.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔢 Enter Market Data")

    col1, col2 = st.columns(2)

    with col1:
        open_price = st.number_input("Open Price ($)", value=1800.00, step=0.01, format="%.2f")
        low_price = st.number_input("Low Price ($)", value=1750.00, step=0.01, format="%.2f")
    with col2:
        high_price = st.number_input("High Price ($)", value=1850.00, step=0.01, format="%.2f")
        volume = st.number_input("Volume", value=100000000.00, step=1000000.0, format="%.2f")

    st.markdown("---")

    # Prediction button
    if st.button("🔮 Predict Next Close Price"):
        try:
            result = predict_next_price(open_price, high_price, low_price, volume)
            st.success(f"📈 **Predicted Close Price:** ${result:.2f}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

    st.markdown("---")

if __name__ == "__main__":
    main()

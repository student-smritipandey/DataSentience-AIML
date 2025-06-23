import streamlit as st
from eligibility_logic import check_eligibility


st.title("💳 Credit Card Eligibility Checker")

age = st.number_input("Enter your age:", min_value=0, step=1)
income = st.number_input("Enter your monthly income (₹):", min_value=0, step=1000)
employment_status = st.selectbox("Select your employment status:", ["Salaried", "Self-Employed", "Student", "Unemployed"])

if st.button("Check Eligibility"):
    eligible, message = check_eligibility(age, income, employment_status)
    if eligible:
        st.success(message)
    else:
        st.error(message)

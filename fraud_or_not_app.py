
import streamlit as st
import pandas as pd
import random
import pickle

# Load model and features (mock or real)
try:
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
except FileNotFoundError:
    model = None
    feature_columns = ['TimeOfDay', 'Location', 'Amount', 'Merchant', 'Item']

# Dummy transactions (funny + realistic examples)
transactions = [
    {'TransactionID': 1, 'TimeOfDay': 3, 'Location': 'Thunder Bay', 'Amount': 1200, 'Merchant': 'Hoverboards R Us', 'Item': '6 Hoverboards'},
    {'TransactionID': 2, 'TimeOfDay': 14, 'Location': 'Toronto', 'Amount': 400, 'Merchant': 'IKEA', 'Item': 'Furniture Haul'},
    {'TransactionID': 3, 'TimeOfDay': 2, 'Location': 'Sudbury', 'Amount': 9999, 'Merchant': 'Crypto Kiosk', 'Item': 'Bitcoin'},
    {'TransactionID': 4, 'TimeOfDay': 17, 'Location': 'Vancouver', 'Amount': 89.95, 'Merchant': 'Tim Hortons', 'Item': '300 Timbits'},
    {'TransactionID': 5, 'TimeOfDay': 12, 'Location': 'Halifax', 'Amount': 15.75, 'Merchant': 'Grocery Store', 'Item': 'Lunch Sandwich'}
]

df = pd.DataFrame(transactions)

st.title("🕵️‍♂️ Fraud or Not? You Decide!")
st.markdown("A fun AI demo based on fake Moneris-like transaction data.")

selected_id = st.selectbox("Select a Transaction ID", df['TransactionID'])
transaction = df[df['TransactionID'] == selected_id].iloc[0]

st.subheader("Transaction Details")
st.write(transaction.to_frame())

user_guess = st.radio("What do YOU think? 🤔", ("Fraud", "Not Fraud"))

# Simple fake model prediction logic if no model is loaded
if model:
    input_data = pd.DataFrame([transaction])[feature_columns]
    prediction = model.predict(input_data)[0]
else:
    # Simulate a prediction for fun
    prediction = 1 if transaction['Amount'] > 1000 or transaction['TimeOfDay'] < 6 else 0

st.subheader("🔍 Model's Prediction:")
st.success("Fraud ✅" if prediction == 1 else "Not Fraud ❎")

# Compare to user's choice
if user_guess == ("Fraud" if prediction == 1 else "Not Fraud"):
    st.balloons()
    st.markdown("**🎉 You matched the AI!**")
else:
    st.markdown("🤖 The AI thinks otherwise... but who's right?")

st.markdown("---")
st.caption("Powered by Streamlit + Fake Data + Your Good Instincts 🧠")

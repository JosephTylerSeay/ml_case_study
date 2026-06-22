import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# -----------------------------
# Load and prepare data
# -----------------------------

loan_df = pd.read_csv("loan_data.csv")

loan_df["fico_base"] = (
    loan_df["fico_range"]
    .str.split("-")
    .str[0]
    .astype(int)
)


X = loan_df[
    [
        "amount_requested",
        "fico_base",
        "loan_length"
    ]
]

y = loan_df["interest_rate"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Helper function
# -----------------------------

def total_interest(principal, apr, months):
    r = apr / 12

    payment = principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)

    return payment * months - principal


# -----------------------------
# Streamlit app
# -----------------------------

st.title("Loan Interest Rate Predictor")

st.write(
    "Enter loan details to estimate an interest rate and total interest charges."
)


fico_base = st.number_input(
    "FICO score",
    min_value=300,
    max_value=850,
    value=700
)

amount_requested = st.number_input(
    "Amount requested ($)",
    min_value=1000,
    value=10000,
    step=1000
)

loan_length = st.selectbox(
    "Loan length",
    sorted(loan_df["loan_length"].unique())
)


user_input = pd.DataFrame({
    "amount_requested": [amount_requested],
    "fico_base": [fico_base],
    "loan_length": [loan_length]
})


predicted_rate = model.predict(user_input)[0]

total_interest_charge = total_interest(
    amount_requested,
    predicted_rate / 100,
    loan_length
)


st.subheader("Prediction Results")

st.metric(
    "Predicted Interest Rate",
    f"{predicted_rate:.2f}%"
)

st.metric(
    "Estimated Total Interest",
    f"${total_interest_charge:,.2f}"
)
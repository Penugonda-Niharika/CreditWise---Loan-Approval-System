import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "creditwise_model.joblib"

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(120deg, #102a43, #176b87); color: white;}
.small-note {color: #64748b; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🏦 CreditWise Loan Approval System</h1>'
    '<p>Explore a machine-learning estimate based on applicant and loan details.</p></div>',
    unsafe_allow_html=True,
)
st.write("")

if not MODEL_PATH.exists():
    st.error("The trained model file is missing.")
    st.markdown("""
    **Set up the model first:**
    1. Put `loan_approval_data.csv` in this project folder.
    2. Install packages from `requirements.txt`.
    3. Run `python train_model.py`.
    4. Run `streamlit run app.py`.
    """)
    st.stop()

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.caption("Enter the applicant details below. Fields are initialized with reasonable example values; replace them with the applicant's actual information.")

with st.form("loan_application"):
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)

    with c1:
        applicant_income = st.number_input("Applicant income", min_value=0.0, value=10000.0, step=500.0)
        coapplicant_income = st.number_input("Co-applicant income", min_value=0.0, value=3000.0, step=500.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        dependents = st.selectbox("Number of dependents", [0, 1, 2, 3], index=1)
        education = st.selectbox("Education level", ["Graduate", "Not Graduate"])

    with c2:
        employment = st.selectbox("Employment status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        marital = st.selectbox("Marital status", ["Married", "Single"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        employer_category = st.selectbox("Employer category", ["Private", "Government", "MNC", "Business", "Unemployed"])
        credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=680, step=1)

    with c3:
        existing_loans = st.number_input("Existing loans", min_value=0, max_value=30, value=1, step=1)
        dti_ratio = st.slider("Debt-to-income ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
        savings = st.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)
        collateral_value = st.number_input("Collateral value", min_value=0.0, value=25000.0, step=1000.0)
        loan_amount = st.number_input("Requested loan amount", min_value=0.0, value=20000.0, step=1000.0)
        loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)

    c4, c5 = st.columns(2)
    with c4:
        loan_purpose = st.selectbox("Loan purpose", ["Car", "Education", "Home", "Personal", "Business"])
    with c5:
        property_area = st.selectbox("Property area", ["Rural", "Semiurban", "Urban"])

    submitted = st.form_submit_button("Estimate loan approval", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment,
        "Age": float(age),
        "Marital_Status": marital,
        "Dependents": float(dependents),
        "Credit_Score": float(credit_score),
        "Existing_Loans": float(existing_loans),
        "DTI_Ratio": float(dti_ratio),
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": float(loan_term),
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education,
        "Gender": gender,
        "Employer_Category": employer_category,
    }], columns=feature_names)

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        approval_probability = float(probabilities[positive_index])

        st.divider()
        st.subheader("Prediction result")
        left, right = st.columns([1, 2])
        with left:
            if prediction == 1:
                st.success("Predicted: Loan likely approved")
            else:
                st.warning("Predicted: Loan likely not approved")
        with right:
            st.metric("Estimated approval probability", f"{approval_probability:.1%}")
            st.progress(max(0.0, min(1.0, approval_probability)))
        st.caption("This is a machine-learning estimate, not a real lending decision or a guarantee. Use only for educational demonstration.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check that the training dataset uses the expected column names and categories.")

st.divider()
st.markdown(
    '<p class="small-note">CreditWise is an educational ML demo. Predictions can be inaccurate and should not be used as the sole basis for real credit decisions.</p>',
    unsafe_allow_html=True,
)

# CreditWise Loan Approval System

A beginner-friendly Streamlit interface for an educational loan approval classification model.

## Files

- `app.py` — Streamlit user interface.
- `train_model.py` — trains a Logistic Regression model and saves the preprocessing + model pipeline.
- `preprocessing.py` — shared feature-engineering function required when loading the saved model.
- `requirements.txt` — Python dependencies.
- `loan_approval_data.csv` — **you must add your original dataset**; it is not included because it was not attached separately.

## 1. Expected dataset columns

The CSV must include:

`Applicant_ID` (optional; ignored), `Applicant_Income`, `Coapplicant_Income`, `Employment_Status`, `Age`, `Marital_Status`, `Dependents`, `Credit_Score`, `Existing_Loans`, `DTI_Ratio`, `Savings`, `Collateral_Value`, `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Property_Area`, `Education_Level`, `Gender`, `Employer_Category`, `Loan_Approved`.

`Loan_Approved` should contain `Yes`/`No` (case-insensitive) or 1/0.

The app exposes the feature columns used by the training script. Categories not seen during training are handled by the one-hot encoder.

## 2. Install dependencies

Open a terminal in this folder:

```bash
python -m pip install -r requirements.txt
```

## 3. Train the model

Copy your original `loan_approval_data.csv` into this folder, then run:

```bash
python train_model.py
```

This creates `creditwise_model.joblib`.

## 4. Run the UI

```bash
streamlit run app.py
```

Streamlit will show a local URL, normally `http://localhost:8501`.

## 5. Deploy with Streamlit Community Cloud

1. Push `app.py`, `preprocessing.py`, `train_model.py`, `requirements.txt`, and the generated `creditwise_model.joblib` to a GitHub repository.
2. In Streamlit Community Cloud, choose that repository and set `app.py` as the app entrypoint.
3. Deploy.

If you prefer not to commit a model artifact, you can instead add the dataset and train during deployment, but this increases startup time and requires the dataset to be available in the deployed repository.

## Notes and limitations

- This is an educational prediction demo, not a real lending decision tool.
- It uses a fresh preprocessing pipeline with median imputation for numeric fields, most-frequent imputation for categorical fields, one-hot encoding, standard scaling, and Logistic Regression.
- Metrics may differ from the notebook because the pipeline avoids fitting preprocessing on the full dataset before the split and uses stratified splitting.
- A model's output is not a guarantee of approval. Do not use this demo for real credit decisions without extensive validation, fairness review, privacy/security controls, and domain oversight.

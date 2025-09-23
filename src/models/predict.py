import os
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "production_model.joblib")

# Load model once
model = joblib.load(MODEL_PATH)

def predict_single_customer(customer_data: dict) -> str:
    # 1️⃣ Convert tenure -> tenure_group
    tenure_bins = [0, 13, 25, 37, 48, 60, 72]
    tenure_labels = ["0-13", "13-25", "25-37", "37-48", "48-60", "60-72"]
    if "tenure" in customer_data:
        tenure_val = customer_data.pop("tenure")
        for i in range(len(tenure_bins)-1):
            if tenure_bins[i] <= tenure_val <= tenure_bins[i+1]:
                customer_data["tenure_group"] = tenure_labels[i]
                break

    # 2️⃣ Convert SeniorCitizen to Yes/No
    if "SeniorCitizen" in customer_data:
        customer_data["SeniorCitizen"] = "Yes" if customer_data["SeniorCitizen"] in [1, "1"] else "No"

    # 3️⃣ Ensure all categorical columns are strings
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure_group', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    for col in categorical_cols:
        if col in customer_data:
            customer_data[col] = str(customer_data[col])

    # 4️⃣ Convert to DataFrame & drop customerID
    df = pd.DataFrame([customer_data])
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # 5️⃣ Predict
    pred = model.predict(df)[0]
    return "Churn" if pred == "Yes" else "No Churn"


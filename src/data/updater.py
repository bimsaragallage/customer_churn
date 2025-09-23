# src/data/updater.py
import pandas as pd
import random
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def append_random_rows():
    df = pd.read_csv(RAW_DATA_PATH)

    new_rows = []
    for _ in range(3):
        tenure = random.randint(1, 72)  # tenure in months
        monthly = round(random.uniform(20, 120), 2)
        total = round(monthly * tenure, 2)

        new_row = {
            "customerID": f"NEW-{random.randint(10000, 99999)}",
            "gender": random.choice(["Male", "Female"]),
            "SeniorCitizen": random.choice([0, 1]),
            "Partner": random.choice(["Yes", "No"]),
            "Dependents": random.choice(["Yes", "No"]),
            "tenure": tenure,
            "PhoneService": random.choice(["Yes", "No"]),
            "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
            "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
            "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
            "DeviceProtection": random.choice(["Yes", "No", "No internet service"]),
            "TechSupport": random.choice(["Yes", "No", "No internet service"]),
            "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
            "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
            "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": random.choice(["Yes", "No"]),
            "PaymentMethod": random.choice([
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ]),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": random.choice(["Yes", "No"])
        }

        new_rows.append(new_row)

    # Append to dataset
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"✅ Appended 3 new rows to {RAW_DATA_PATH}")

if __name__ == "__main__":
    append_random_rows()

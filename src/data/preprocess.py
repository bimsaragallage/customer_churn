# src/data/preprocess.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(dataset_filename="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    # Determine project root (two levels up from this script)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    raw_data_path = os.path.join(project_root, "datasets", "raw", dataset_filename)

    # Load dataset
    data = pd.read_csv(raw_data_path)

    # -----------------------------
    # 1️⃣ Clean 'TotalCharges' column
    # -----------------------------
    col = "TotalCharges"
    col_values = data[col].astype(str).str.strip()
    is_numeric = col_values.str.replace(".", "", 1).str.isnumeric()
    data = data[is_numeric].copy()

    numeric_cols = ["MonthlyCharges", "TotalCharges"]
    data[numeric_cols] = data[numeric_cols].astype("float64")

    # -----------------------------
    # 2️⃣ Fixed tenure bins
    # -----------------------------
    bins = [0, 13, 25, 37, 48, 60, 72]
    labels = ["0-13", "13-25", "25-37", "37-48", "48-60", "60-72"]
    data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels, include_lowest=True).astype('object')

    # -----------------------------
    # 3️⃣ Convert 'SeniorCitizen' to 'No'/'Yes'
    # -----------------------------
    data['SeniorCitizen'] = data['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).astype('object')
    data.drop('tenure', axis=1, inplace=True)

    # -----------------------------
    # 4️⃣ Remove outliers
    # -----------------------------
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    # -----------------------------
    # 5️⃣ Split features and target
    # -----------------------------
    customer_id_col = "customerID"
    target_col = "Churn"
    X = data.drop([customer_id_col, target_col], axis=1)
    y = data[target_col]

    # -----------------------------
    # 6️⃣ Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 7️⃣ Save cleaned raw datasets
    # -----------------------------
    train_dir = os.path.join(project_root, "datasets", "train")
    test_dir  = os.path.join(project_root, "datasets", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Churn'] = y_train
    train_df.to_csv(os.path.join(train_dir, "train.csv"), index=False)

    test_df = X_test.copy()
    test_df['Churn'] = y_test
    test_df.to_csv(os.path.join(test_dir, "test.csv"), index=False)

    print("Preprocessing completed. Cleaned raw train/test CSVs saved.")

if __name__ == "__main__":
    preprocess_data()

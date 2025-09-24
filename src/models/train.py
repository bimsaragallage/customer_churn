# src/models/train.py
import os
import time
import platform
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn

# -----------------------------
# Config
# -----------------------------
TARGET_COL = "Churn"

# Project root (repo root, 2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Dataset & model paths
TRAIN_PATH = os.path.join(PROJECT_ROOT, "datasets", "train", "train.csv")
TEST_PATH  = os.path.join(PROJECT_ROOT, "datasets", "test", "test.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_smoteenn_model.joblib")
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(MLRUNS_PATH, exist_ok=True)

# Portable MLflow tracking URI
if platform.system() == "Windows":
    MLFLOW_URI = f"file:///{os.path.abspath(MLRUNS_PATH).replace(os.sep, '/')}"
else:
    MLFLOW_URI = f"file://{os.path.abspath(MLRUNS_PATH)}"

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_tracking_uri(MLFLOW_URI)

# Explicitly create/set experiment
EXPERIMENT_NAME = "CustomerChurn_RF"
mlflow.set_experiment(EXPERIMENT_NAME)
print("🔗 Tracking URI:", mlflow.get_tracking_uri())
print("🧪 Experiment:", EXPERIMENT_NAME)

# -----------------------------
# Feature columns
# -----------------------------
NUMERIC_COLS = ["MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure_group',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# -----------------------------
# Load train/test datasets
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train, y_train = train_df.drop(TARGET_COL, axis=1), train_df[TARGET_COL]
X_test,  y_test  = test_df.drop(TARGET_COL, axis=1), test_df[TARGET_COL]

# -----------------------------
# Preprocessor
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
    ]
)

# -----------------------------
# Full pipeline
# -----------------------------
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smoteenn', SMOTEENN(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# -----------------------------
# Hyperparameter grid
# -----------------------------
PARAM_GRID = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__criterion': ['gini', 'entropy']
}

# -----------------------------
# Train function
# -----------------------------
def train_model():
    print("🚀 Training Random Forest with SMOTE+ENN and GridSearchCV...")

    with mlflow.start_run(run_name="RandomForest_SMOTEENN"):
        # Log dataset/feature info
        mlflow.log_param("numeric_features", NUMERIC_COLS)
        mlflow.log_param("categorical_features", CATEGORICAL_COLS)

        # Hyperparameter tuning
        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRID,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = classification_report(
            y_test, y_pred, output_dict=True
        )['macro avg']['f1-score']

        # Log best params & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("train_time_seconds", train_time)

        # Save model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )
        joblib.dump(best_model, MODEL_PATH)

        # Print results
        print("\n✅ Training complete!")
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"Training Time: {train_time:.2f} sec")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print(f"\n💾 Model saved at {MODEL_PATH}")
        print(f"📂 MLflow run stored in: {MLRUNS_PATH}")

# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    train_model()

# src/models/train.py
import os
import time
import pandas as pd
import joblib
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "datasets", "train", "train.csv")
TEST_PATH  = os.path.join(PROJECT_ROOT, "datasets", "test", "test.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_smoteenn_model.joblib")
mlruns_path = os.path.join(PROJECT_ROOT, "mlruns")

# Tracking URI
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("CustomerChurn_RF")   # group runs into experiment
print("Tracking URI:", mlflow.get_tracking_uri())

NUMERIC_COLS = ["MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure_group',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# -----------------------------
# Load cleaned train/test
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop(TARGET_COL, axis=1)
y_train = train_df[TARGET_COL]
X_test  = test_df.drop(TARGET_COL, axis=1)
y_test  = test_df[TARGET_COL]

# -----------------------------
# Preprocessor inside pipeline
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
# Hyperparameter tuning
# -----------------------------
PARAM_GRID = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__criterion': ['gini', 'entropy']
}

def train_model():
    print("Training Random Forest with SMOTE+ENN and GridSearchCV...")

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_SMOTEENN"):
        # Log input feature info
        mlflow.log_param("numeric_features", NUMERIC_COLS)
        mlflow.log_param("categorical_features", CATEGORICAL_COLS)

        start_time = time.time()
        grid_search = GridSearchCV(
            pipeline,
            PARAM_GRID,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        best_model = grid_search.best_estimator_

        # -----------------------------
        # Evaluate on test set
        # -----------------------------
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']

        # -----------------------------
        # Log hyperparameters and metrics
        # -----------------------------
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("train_time_seconds", train_time)

        # -----------------------------
        # Log model + register
        # -----------------------------
        model_name = "CustomerChurn_RF"
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name=model_name
        )

        # -----------------------------
        # Save model locally too
        # -----------------------------
        #os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        #joblib.dump(best_model, MODEL_PATH)

        print(f"✅ Training complete!")
        print("Accuracy:", acc)
        print("F1 Macro:", f1_macro)
        print("Training Time (s):", round(train_time, 2))
        print("Best Params:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print(f"Model saved locally at {MODEL_PATH} and logged to MLflow.")

if __name__ == "__main__":
    train_model()
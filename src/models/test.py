# src/models/test.py
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# -----------------------------
# Paths & MLflow config
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

mlruns_path = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri("file:./mlruns")

MODEL_NAME = "CustomerChurn_RF"
client = MlflowClient()

# -----------------------------
# Promote best model to Production
# -----------------------------
def promote_best_model():
    print("🔍 Checking runs to find best model...")

    # Use the experiment where your training runs are logged
    experiment = mlflow.get_experiment_by_name("CustomerChurn_RF")
    if experiment is None:
        raise Exception("Experiment 'CustomerChurn_RF' not found!")
    experiment_id = experiment.experiment_id

    # Get all runs sorted by F1 score
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=5
    )

    if not runs:
        print("❌ No runs found.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics["f1_macro"]
    print(f"✅ Best run {best_run_id} with F1: {best_f1:.4f}")

    # Register model if not exists
    try:
        client.create_registered_model(MODEL_NAME)
        print(f"📦 Registered new model {MODEL_NAME}")
    except Exception:
        print(f"ℹ️ Model {MODEL_NAME} already registered.")

    # Register new version
    model_uri = f"runs:/{best_run_id}/model"
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=best_run_id
    )
    version = mv.version
    print(f"📌 Created version {version} for model {MODEL_NAME}")

    # Transition to Production (archive old versions)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 Promoted version {version} to Production.")

    # -----------------------------
    # Download Production model locally
    # -----------------------------
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    prod_version = versions[0]
    prod_model_uri = f"runs:/{prod_version.run_id}/model"

    prod_model = mlflow.sklearn.load_model(prod_model_uri)
    local_path = os.path.join(MODEL_DIR, "production_model.joblib")
    joblib.dump(prod_model, local_path)
    print(f"💾 Production model saved locally at {local_path}")

# -----------------------------
# Run promotion
# -----------------------------
if __name__ == "__main__":
    promote_best_model()

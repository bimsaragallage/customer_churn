from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict import predict_single_customer

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict customer churn using trained Random Forest model",
    version="1.0.0"
)

# -----------------------------
# Request schema (input validation)
# -----------------------------
class CustomerData(BaseModel):
    customerID: str | None = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running."}

@app.post("/predict/")
def api_predict(customer: CustomerData):
    """
    Accepts a single customer's data in JSON format,
    returns churn prediction ("Churn" or "No Churn").
    """
    try:
        result = predict_single_customer(customer.dict())
        return {"Prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
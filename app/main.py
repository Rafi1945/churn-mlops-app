import pandas as pd
from fastapi import FastAPI

from app.schemas import ChurnInput, ChurnPredictionResponse
from app.model_loader import load_model, load_metrics


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-style ML API for customer churn prediction",
    version="1.0.0"
)

model = load_model()
metrics = load_metrics()


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "ChurnClassifier",
        "model_type": "XGBClassifier Pipeline",
        "metrics": metrics
    }


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(data: ChurnInput):
    input_df = pd.DataFrame([
        {
            "Age": data.Age,
            "Gender": data.Gender,
            "Tenure": data.Tenure,
            "Usage Frequency": data.Usage_Frequency,
            "Support Calls": data.Support_Calls,
            "Payment Delay": data.Payment_Delay,
            "Subscription Type": data.Subscription_Type,
            "Contract Length": data.Contract_Length,
            "Total Spend": data.Total_Spend,
            "Last Interaction": data.Last_Interaction
        }
    ])

    churn_probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(churn_probability >= 0.5)

    prediction_label = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "churn_probability": round(churn_probability, 4)
    }
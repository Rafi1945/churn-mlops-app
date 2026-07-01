from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "ChurnClassifier"


def test_predict_endpoint():
    payload = {
        "Age": 35,
        "Gender": "Female",
        "Tenure": 12,
        "Usage_Frequency": 15,
        "Support_Calls": 3,
        "Payment_Delay": 10,
        "Subscription_Type": "Premium",
        "Contract_Length": "Monthly",
        "Total_Spend": 550.75,
        "Last_Interaction": 8
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "prediction_label" in result
    assert "churn_probability" in result

    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["Churn", "No Churn"]
    assert 0.0 <= result["churn_probability"] <= 1.0
from pydantic import BaseModel, Field


class ChurnInput(BaseModel):
    Age: int = Field(..., example=35)
    Gender: str = Field(..., example="Female")
    Tenure: int = Field(..., example=12)
    Usage_Frequency: int = Field(..., example=15)
    Support_Calls: int = Field(..., example=3)
    Payment_Delay: int = Field(..., example=10)
    Subscription_Type: str = Field(..., example="Premium")
    Contract_Length: str = Field(..., example="Monthly")
    Total_Spend: float = Field(..., example=550.75)
    Last_Interaction: int = Field(..., example=8)


class ChurnPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: float
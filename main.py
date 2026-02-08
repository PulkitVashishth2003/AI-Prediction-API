from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="AI Selection Prediction API")

# Load trained model (make sure model.pkl exists in same folder)
model = joblib.load("model.pkl")

# Input schema
class CandidateInput(BaseModel):
    age: int
    experience: int
    score: int


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running successfully"}

# Prediction endpoint
@app.post("/predict")
def predict_selection(candidate: CandidateInput):
    data = pd.DataFrame(
        [[candidate.age, candidate.experience, candidate.score]],
        columns=["age", "experience", "score"]
    )

    prediction = model.predict(data)

    result = "Selected" if prediction[0] == 1 else "Rejected"

    return {
        "input": candidate,
        "prediction": result
    }

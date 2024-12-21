from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Delivery Prediction API",
             description="API for predicting delivery outcomes based on shipment data")

# Load the model and label encoders
try:
    model = joblib.load("gradient_boosting_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    raise Exception(f"Error loading model files: {str(e)}")

# Define input data model
class DeliveryData(BaseModel):
    Origin: str
    Destination: str
    Vehicle_Type: str
    Distance_km: float
    Weather_Conditions: str
    Traffic_Conditions: str
    Planned_Delay_days: float
    Actual_Delay_days: float
    Delay_Ratio: float
    Severe_Weather: int
    Shipment_Day: int
    Shipment_Month: int
    Shipment_Year: int
    Planned_Delivery_Day: int
    Planned_Delivery_Month: int
    Planned_Delivery_Year: int
    Actual_Delivery_Day: int
    Actual_Delivery_Month: int
    Actual_Delivery_Year: int

class PredictionResponse(BaseModel):
    prediction: float
    status: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(delivery_data: DeliveryData):
    try:
        # Convert input data to DataFrame
        input_data = {
            'Origin': [delivery_data.Origin],
            'Destination': [delivery_data.Destination],
            'Vehicle Type': [delivery_data.Vehicle_Type],
            'Distance (km)': [delivery_data.Distance_km],
            'Weather Conditions': [delivery_data.Weather_Conditions],
            'Traffic Conditions': [delivery_data.Traffic_Conditions],
            'Planned Delay (days)': [delivery_data.Planned_Delay_days],
            'Actual Delay (days)': [delivery_data.Actual_Delay_days],
            'Delay Ratio': [delivery_data.Delay_Ratio],
            'Severe Weather': [delivery_data.Severe_Weather],
            'Shipment Day': [delivery_data.Shipment_Day],
            'Shipment Month': [delivery_data.Shipment_Month],
            'Shipment Year': [delivery_data.Shipment_Year],
            'Planned Delivery Day': [delivery_data.Planned_Delivery_Day],
            'Planned Delivery Month': [delivery_data.Planned_Delivery_Month],
            'Planned Delivery Year': [delivery_data.Planned_Delivery_Year],
            'Actual Delivery Day': [delivery_data.Actual_Delivery_Day],
            'Actual Delivery Month': [delivery_data.Actual_Delivery_Month],
            'Actual Delivery Year': [delivery_data.Actual_Delivery_Year]
        }
        
        input_df = pd.DataFrame(input_data)

        # Encode categorical columns
        categorical_columns = ['Origin', 'Destination', 'Vehicle Type', 
                             'Weather Conditions', 'Traffic Conditions']
        
        for col in categorical_columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value for {col}. Must be one of: {list(le.classes_)}"
                    )

        # Ensure column order matches the training data
        columns_order = [
            'Origin', 'Destination', 'Vehicle Type', 'Distance (km)', 
            'Weather Conditions', 'Traffic Conditions', 'Planned Delay (days)', 
            'Actual Delay (days)', 'Delay Ratio', 'Severe Weather', 
            'Shipment Day', 'Shipment Month', 'Shipment Year', 
            'Planned Delivery Day', 'Planned Delivery Month', 'Planned Delivery Year',
            'Actual Delivery Day', 'Actual Delivery Month', 'Actual Delivery Year'
        ]
        
        input_df = input_df[columns_order]

        # Make prediction
        prediction = model.predict(input_df)[0]

        return PredictionResponse(
            prediction=float(prediction),
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
async def root():
    return {"message": "Delivery Prediction API is running. Use /predict endpoint for predictions."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# Delivery Prediction API

A FastAPI-based web service for predicting delivery outcomes (e.g., delayed or not) based on shipment data. The service uses a pre-trained Gradient Boosting Classifier model.
### Features

    Predict Delivery Outcomes: Accepts shipment data and returns a prediction (0: Not Delayed, 1: Delayed).
    Error Handling: Provides informative error messages for invalid or missing inputs.
    FastAPI Framework: Lightweight, fast, and easy-to-extend API framework.

### Setup Instructions

    Install Dependencies: pip install fastapi uvicorn pydantic joblib pandas scikit-learn

Run the API:

    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    then after running paste { http://127.0.0.1:8000/docs#/default/predict_predict_post } this on your browser

    Endpoints:
        GET /: Basic health check.
        POST /predict: Predicts the delivery outcome based on input JSON.

Example Input (for /predict)

{
    "Origin": "A",
    "Destination": "B",
    "Vehicle_Type": "Truck",
    "Distance_km": 500.0,
    "Weather_Conditions": "Clear",
    "Traffic_Conditions": "Moderate",
    "Planned_Delay_days": 2.0,
    "Actual_Delay_days": 3.0,
    "Delay_Ratio": 1.5,
    "Severe_Weather": 0,
    "Shipment_Day": 12,
    "Shipment_Month": 8,
    "Shipment_Year": 2023,
    "Planned_Delivery_Day": 14,
    "Planned_Delivery_Month": 8,
    "Planned_Delivery_Year": 2023,
    "Actual_Delivery_Day": 15,
    "Actual_Delivery_Month": 8,
    "Actual_Delivery_Year": 2023
}

Example Response

{
    "prediction": 1.0,
    "status": "success"
}

Notes

    Ensure gradient_boosting_model.pkl and label_encoders.pkl are in the root directory.
    Categorical fields must match the values used during training.

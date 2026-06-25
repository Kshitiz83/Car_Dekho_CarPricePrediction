from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize FastAPI application
app = FastAPI(title="Used Car Price Predictor API")

# 2. Load the trained components
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("car_scaler.pkl")

# Get the exact number of features the model expects
num_features = model.n_features_in_

# 3. Create a welcoming Home Route
@app.get("/")
def home():
    return {"message": "Welcome to the Used Car Price Predictor API! Head over to /docs to test it out."}

# 4. Define the input data validation schema
# To keep testing simple, we accept the primary 4 continuous values
class CarFeatures(BaseModel):
    vehicle_age: int
    km_driven: int
    mileage: float
    engine: int

# 5. Create the Prediction Endpoint
@app.post("/predict")
def predict_car_price(data: CarFeatures):
    # Setup a full-sized array of zeros matching your exact training feature count (44 columns)
    input_array = np.zeros((1, num_features))
    
    # Map the primary inputs into the first 4 feature slots
    input_array[0, 0] = data.vehicle_age
    input_array[0, 1] = data.km_driven
    input_array[0, 2] = data.mileage
    input_array[0, 3] = data.engine
    
    # CRITICAL: Scale the inputs using our saved scaler!
    scaled_input = scaler.transform(input_array)
    
    # Make the log prediction and convert it back to actual currency
    log_pred = model.predict(scaled_input)
    actual_price = np.expm1(log_pred)
    
    return {"predicted_selling_price": f"${actual_price[0]:,.2f}"}

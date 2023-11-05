import uvicorn
from fastapi import FastAPI

from basket_model import BasketModel
from feature_store import FeatureStore
from pydantic import BaseModel

class User_id(BaseModel):
    user_id: str
# Create an instance of FastAPI
app = FastAPI()

# Load Model and Feature Store
loaded_model = BasketModel()
loaded_feature_store = FeatureStore()

print(loaded_feature_store)

# Define a route for the root URL ("/")
@app.get("/")
def read_root():
    return {"message": "API for predicting if a particular user will be interested in one of our products.\
            Returns 1 if we expect the user to be interested in the product and 0 if we don't."}

@app.get("/status")
def read_status():
    return {"status": 200}

@app.post("/predict")
def predict_outcome(data:User_id):
    features = loaded_feature_store.get_features(data)
    prediction = loaded_model.predict(features)
    return {"prediction": prediction}

# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import uvicorn
from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse
import time

from basket_model import BasketModel
from feature_store import FeatureStore
from pydantic import BaseModel
import logging
from exceptions import PredictionException, UserNotFoundException


#Set logging
logging.basicConfig(
    filename="src/module_6/api_log.txt",
    filemode="a",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

class User_id(BaseModel):
    user_id: str

# Create an instance of FastAPI
app = FastAPI()

# Load Model and Feature Store
loaded_model = BasketModel()
loaded_feature_store = FeatureStore()

#Logging of the model
@app.middleware("http")
async def api_logging(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = "{0:.2f} ms.".format(process_time)

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk
    log_message = {
        "host": request.url.hostname,
        "endpoint": request.url.path,
        "response": response_body.decode(),
        "latency" : formatted_process_time
    }
    logger.debug(log_message)
    return Response(content=response_body, status_code=response.status_code, 
        headers=dict(response.headers), media_type=response.media_type)

#Root
@app.get("/")
def read_root():
    return {"message": "API for predicting if a particular user will be interested in one of our products.\
            Returns 1 if we expect the user to be interested in the product and 0 if we don't."}

#Status
@app.get("/status")
def read_status():
    return JSONResponse(status_code=200)

#Predict Method
@app.post("/predict")
async def predict_outcome(data: User_id):
    features = loaded_feature_store.get_features(data.user_id)
    prediction = loaded_model.predict(features)
    return {"prediction": prediction.tolist()[0]}

#Exception Handler for PredictionException error
@app.exception_handler(PredictionException)
async def prediction_exception_handler(request: Request, exc: PredictionException):
    return JSONResponse(
        status_code=404,
        content={"message": f"{exc.name}"},
    )

#Exception Handler for UserNotFoundException error
@app.exception_handler(UserNotFoundException)
async def user_not_found_handler(request: Request, exc: UserNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"message": f"{exc.name}"},
    )

# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
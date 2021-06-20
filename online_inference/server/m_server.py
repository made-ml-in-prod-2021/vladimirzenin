from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import json

import sys
sys.path.insert(0, '../scripts')
from predict import get_predict_json
from data_models import JsonItem, JsonArr
import datetime


start_date = datetime.datetime.now()
app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"message": "Validation error! Please check documentation."}
    )


@app.post("/predict_single/",
    description="Predict single item",
    response_model=str)
async def get_predict_single(input_data: JsonItem) -> str:
    arr = JsonArr(data=[input_data.__dict__])
    answ = get_predict_json(arr)
    return answ


@app.post("/predict_batch/",
    description="Predict batch item (list of item).",
    response_model=str)
async def get_predict_batch(input_data: JsonArr) -> str:
    answ = get_predict_json(input_data)
    return answ


@app.get("/readiness")
def readiness() -> bool:
    curr_date = datetime.datetime.now()
    if (curr_date - start_date).seconds > 20:
        return True
    else:
        return False


@app.get("/liveness")
def liveness() -> bool:
    curr_date = datetime.datetime.now()
    if (curr_date - start_date).seconds > 60:
        sys.exit()
    else:
        return True


if __name__ == "__main__":
    uvicorn.run("m_server:app", host="0.0.0.0", port=8050, log_level="info")

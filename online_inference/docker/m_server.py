from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
import json

import sys
sys.path.insert(0, '../../ml_project/scripts')
from predict import get_predict_json
from data_models import JsonItem, JsonArr


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
    #try:
    arr = JsonArr(data=[input_data.__dict__])
    answ = get_predict_json(arr)
    #except IndexError:
    #    raise HTTPException(404, "Phrase list is empty")
    return answ


@app.post("/predict_batch/",
    description="Predict batch item (list of item).",
    response_model=str)
async def get_predict_batch(input_data: JsonArr) -> str:
    answ = get_predict_json(input_data)
    return answ


if __name__ == "__main__":
    uvicorn.run("m_server:app", host="127.0.0.1", port=8050, log_level="info")

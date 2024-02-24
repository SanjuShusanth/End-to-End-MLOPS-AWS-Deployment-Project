import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from Shipment import model_input
import os
import joblib
from MLProject.pipeline.prediction import PredictionPipeline, Custom_data 
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory='html_directory')


@app.get('/train')
async def training():
    os.system('python main.py')
    return "Training Successful"

@app.get("/home/", response_class=HTMLResponse)
async def hello(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict/', response_class=HTMLResponse)
async def prediction(item: model_input, request: Request):

    test_data = (
        item.Drop_point,
        item.Shipment_Mode,
        item.Dosage_Form,
        item.Line_Item_Quantity,
        item.Pack_Price,
        item.Unit_Price,
        item.Weight,
        item.Freight_Cost,
        item.Line_Item_Insurance,
        item.Delivery_Status,
        item.Pickup_Point
    )

    pipeline = PredictionPipeline()

    pred = pipeline.predict(test_data)

    Results = round(pred[0],2)

    return templates.TemplateResponse('results.html',  {'request': Request, 'result': Results}) 



if __name__=="__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
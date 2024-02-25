import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from Shipment import model_input
from src.MLProject.pipeline.prediction import PredictionPipeline
import pandas as pd
import os

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
async def prediction(request: Request,
    Drop_point: str = Form(...),
    Shipment_Mode: str = Form(...),
    Dosage_Form: str = Form(...),
    Line_Item_Quantity: str = Form(...),
    Pack_Price: str = Form(...),
    Unit_Price: str = Form(...),
    Weight: str = Form(...),
    Freight_Cost: str = Form(...),
    Line_Item_Insurance: str = Form(...),
    Delivery_Status: str = Form(...),
    Pickup_Point: str = Form(...)
):
    test_data = [Drop_point, Shipment_Mode, Dosage_Form, Line_Item_Quantity,
                 Pack_Price, Unit_Price, Weight, Freight_Cost,
                 Line_Item_Insurance, Delivery_Status, Pickup_Point]

    columns = ['Drop_point', 'Shipment_Mode', 'Dosage_Form', 'Line_Item_Quantity',
               'Pack_Price', 'Unit_Price', 'Weight', 'Freight_Cost',
               'Line_Item_Insurance', 'Delivery_Status', 'Pickup_Point']

    input_df = pd.DataFrame([test_data], columns=columns)
    pipeline = PredictionPipeline()
    pred = pipeline.predict(input_df)
    Results = round(pred[0], 2)
    return templates.TemplateResponse('results.html', {'request': request, 'result': Results})

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)


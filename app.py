import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from Shipment import model_input
from src.MLProject.pipeline.prediction import PredictionPipeline
import pandas as pd
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import Depends
from MLProject import logger
#from enum import Enum

app = FastAPI()

app.mount("/static", StaticFiles(directory="html_directory/static"), name="static")

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
        Drop_point:  str = Form(...),
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
        dropdown_mapping = {
            "Côte d'Ivoire": "Côte d'Ivoire",
            "Vietnam": "Vietnam",
            "Nigeria": "Nigeria",
            "Zambia": "Zambia",
            "Tanzania": "Tanzania",
            "Rwanda": "Rwanda",
            "Haiti": "Haiti",
            "Zimbabwe": "Zimbabwe",
            "Ethiopia": "Ethiopia",
            "Others": "Others",
            "South Africa": "South Africa",
            "Guyana": "Guyana",
            "Mozambique": "Mozambique",
            "Kenya": "Kenya",
            "Uganda": "Uganda",
            "South Sudan": "South Sudan",
            "Congo, DRC": "Congo, DRC",
            # Add mappings for other dropdowns
        }

        drop_point_value = dropdown_mapping.get(Drop_point, Drop_point)
        shipment_mode_value = dropdown_mapping.get(Shipment_Mode, Shipment_Mode)
        dosage_form_value = dropdown_mapping.get(Dosage_Form, Dosage_Form)
        delivery_status_value = dropdown_mapping.get(Delivery_Status, Delivery_Status)
        pickup_point_value = dropdown_mapping.get(Pickup_Point, Pickup_Point)

        test_data = [drop_point_value, shipment_mode_value, dosage_form_value, Line_Item_Quantity,
                    Pack_Price, Unit_Price, Weight, Freight_Cost,
                    Line_Item_Insurance, delivery_status_value, pickup_point_value]

        columns = ['Drop_point', 'Shipment_Mode', 'Dosage_Form', 'Line_Item_Quantity',
                'Pack_Price', 'Unit_Price', 'Weight', 'Freight_Cost',
                'Line_Item_Insurance', 'Delivery_Status', 'Pickup_Point']

        input_df = pd.DataFrame([test_data], columns=columns)
        pipeline = PredictionPipeline()
        pred = pipeline.predict(input_df)
        Results = round(pred[0])
        return templates.TemplateResponse('results.html', {'request': request, 'result': Results})

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)


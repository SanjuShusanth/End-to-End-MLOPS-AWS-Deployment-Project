import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path
import pickle
from MLProject.utils.common import load_object
from MLProject import logger

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        try:
            preprocess_obj = Path(r'C:\Users\Sanju\WORKSPACE\End-to-End-MLOPS-AWS-Deployment-Project\artifacts\models\preprocessor.pkl')
            model_path = Path(r'C:\Users\Sanju\WORKSPACE\End-to-End-MLOPS-AWS-Deployment-Project\artifacts\models\model.pkl')
            
            preprocessor = load_object(preprocess_obj)
            model = load_object(model_path)

            data_scalled = preprocessor.transform(pd.DataFrame([[data]], columns=['Drop_point', 'Shipment_Mode', 'Dosage_Form', 'Line_Item_Quantity', 'Pack_Price',
                             'Unit_Price', 'Weight', 'Freight_Cost', 'Line_Item_Insurance', 'Delivery_Status',
                             'Pickup_Point']))

            pred =  model.predict(data_scalled)

            return pred

        except Exception as e:
            logger.info('Exception occurred in prediction')
            raise e
        
class Custom_data:
    def __init__(self,
                 Drop_point: str,
                 Shipment_Mode: str,
                 Dosage_Form: str,
                 Line_Item_Quantity: int,
                 Pack_Price: float,
                 Unit_Price: float,
                 Weight: float,
                 Freight_Cost: float,
                 Line_Item_Insurance: float,
                 Delivery_Status: str,
                 Pickup_Point: str):
        
        self.Drop_point = Drop_point
        self.Shipment_Mode = Shipment_Mode
        self.Dosage_Form = Dosage_Form
        self.Line_Item_Quantity = Line_Item_Quantity
        self.Pack_Price = Pack_Price
        self.Unit_Price = Unit_Price
        self.Weight = Weight
        self.Freight_Cost = Freight_Cost
        self.Line_Item_Insurance = Line_Item_Insurance
        self.Delivery_Status = Delivery_Status
        self.Pickup_Point = Pickup_Point


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Drop_point':[self.Drop_point],
                'Shipment_Mode':[self.Shipment_Mode],
                'Dosage_Form':[self.Dosage_Form],
                'Line_Item_Quantity':[self.Line_Item_Quantity],
                'Pack_Price':[self.Pack_Price],
                'Unit_Price':[self.Unit_Price],
                'Weight':[self.Weight],
                'Freight_Cost':[self.Freight_Cost],
                'Line_Item_Insurance':[self.Line_Item_Insurance],
                'Delivery_Status':[self.Delivery_Status],
                'Pickup_Point':[self.Pickup_Point]
            }

            df = pd.DataFrame(custom_data_input_dict)

            return df
        
        except Exception as e:
            logger.info('Exception occured in custom_data_input')
            raise e
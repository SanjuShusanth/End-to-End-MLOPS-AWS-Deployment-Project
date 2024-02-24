from pydantic import BaseModel


class model_input(BaseModel):
    Drop_point : object
    Shipment_Mode: object
    Dosage_Form: object
    Line_Item_Quantity: int
    Pack_Price: float
    Unit_Price: float
    Weight: float
    Freight_Cost: float
    Line_Item_Insurance: float
    Delivery_Status: object
    Pickup_Point: object
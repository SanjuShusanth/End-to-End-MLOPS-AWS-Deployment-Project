import os
from MLProject import logger
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from MLProject.entity.config_entity import DataTransformationConfig
from MLProject.utils.common import save_object
import pandas as pd
import numpy as np
import pickle



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformation_object(self):
        try:
            Nominal_categorical_cols = ['Drop_point','Dosage_Form','Pickup_Point']
            Ord_categorical_cols = ['Delivery_Status', 'Shipment_Mode']
            Numerical_cols = ['Line_Item_Quantity','Pack_Price','Unit_Price','Weight','Freight_Cost']

            delivery_status = ['Delivered Early', 'On time', 'Delayed']
            shipment_mode = ['Air', 'Truck', 'Air Charter', 'Ocean']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )

            ord_cat_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[delivery_status, shipment_mode]))
                ]
            )


            nom_cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, Numerical_cols),
                ('ord_cat_pipeline', ord_cat_pipeline, Ord_categorical_cols),
                ('nom_cat_pipeline', nom_cat_pipeline, Nominal_categorical_cols)

            ])

            return preprocessor

        except Exception as e:
            raise e
        
    
    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            train_df['Drop_point'].replace("Côte d'Ivoire", "Côte d Ivoire")
            test_df['Drop_point'].replace("Côte d'Ivoire", "Côte d Ivoire")

            input_feature_train_df = train_df.drop(columns=['Line_Item_Value'])
            target_feature_train_df = train_df['Line_Item_Value']

            input_feature_test_df = test_df.drop(columns=['Line_Item_Value'])
            target_feature_test_df = test_df['Line_Item_Value']

            preprocessing_obj = self.get_data_transformation_object()

            input_feature_train_arr = pd.DataFrame(preprocessing_obj.fit_transform(input_feature_train_df), columns=preprocessing_obj.get_feature_names_out())
            input_feature_test_arr = pd.DataFrame(preprocessing_obj.transform(input_feature_test_df), columns=preprocessing_obj.get_feature_names_out())

            train_arr = pd.concat([input_feature_train_arr, target_feature_train_df], axis=1)
            test_arr = pd.concat([input_feature_test_arr, target_feature_test_df], axis=1)

            train_arr.columns=['Line_Item_Quantity', 'Pack_Price', 'Unit_Price', 'Weight', 'Freight_Cost', 'Delivery_Status', 'Shipment_Mode', "Drop_point_Côte d Ivoire", 'Drop_point_Ethiopia', 'Drop_point_Guyana', 'Drop_point_Haiti', 'Drop_point_Kenya', 'Drop_point_Mozambique', 'Drop_point_Nigeria', 'Drop_point_Others', 'Drop_point_Rwanda', 'Drop_point_South Africa', 'Drop_point_South_Sudan', 'Drop_point_Tanzania', 'Drop_point_Uganda', 'Drop_point_Vietnam', 'Drop_point_Zambia', 
                               'Drop_point_Zimbabwe', 'Dosage_Form_Tablet', 'Dosage_Form_Tablet - FDC', 'Dosage_Form_Test kit', 'Pickup_Point_Others', 'Line_Item_Value']

            test_arr.columns=['Line_Item_Quantity', 'Pack_Price', 'Unit_Price', 'Weight', 'Freight_Cost', 'Delivery_Status', 'Shipment_Mode', "Drop_point_Côte d Ivoire", 'Drop_point_Ethiopia', 'Drop_point_Guyana', 'Drop_point_Haiti', 'Drop_point_Kenya', 'Drop_point_Mozambique', 'Drop_point_Nigeria', 'Drop_point_Others', 'Drop_point_Rwanda', 'Drop_point_South Africa', 'Drop_point_South_Sudan', 'Drop_point_Tanzania', 'Drop_point_Uganda', 'Drop_point_Vietnam', 'Drop_point_Zambia', 
                              'Drop_point_Zimbabwe', 'Dosage_Form_Tablet', 'Dosage_Form_Tablet - FDC', 'Dosage_Form_Test kit', 'Pickup_Point_Others', 'Line_Item_Value']
        
            train_arr.to_csv(os.path.join(self.config.root_dir, 'trans_train.csv'), index=False)
            test_arr.to_csv(os.path.join(self.config.root_dir, 'trans_test.csv'), index=False)

            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_path
            )
        except Exception as e:
            print("Exception occurred during data transformation:", e)
            raise e
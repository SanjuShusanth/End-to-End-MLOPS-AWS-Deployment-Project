import os
import pandas as pd
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.MLProject.utils.common import save_json
from pathlib import Path
from src.MLProject.entity.config_entity import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        score = r2_score(actual, pred)

        return rmse, mae, score
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_quantities = model.predict(test_x)
            (rmse, mae, score) = self.eval_metrics(test_y, predicted_quantities)

            # saving metric as local

            scores = {'rmse': rmse, 'mae': mae, 'r2':score}
            save_json(path= Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('score', score)
            mlflow.log_metric('mae', mae)

            if tracking_url_type_store != 'file':


                mlflow.sklearn.log_model(model, 'model', registered_model_name='XgboostRegressionModel')

            else:
                mlflow.sklearn.log_model(model, 'model')
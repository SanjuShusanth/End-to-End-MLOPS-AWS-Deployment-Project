import pandas as pd
import os
from src.MLProject import logger
from xgboost import XGBRegressor
import pickle
from src.MLProject.utils.common import save_object
from src.MLProject.entity.config_entity import ModelTrainerConfig
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        xgb = XGBRegressor(max_depth = self.config.max_depth, learning_rate = self.config.learning_rate, 
                           n_estimators=self.config.n_estimators, random_state=42)
        
        xgb.fit(train_x, train_y)

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))

        save_object(
            file_path = self.config.model_path,
            obj = xgb
        )

        return self.config.model_path
from MLProject.config.configuration import ConfigurationManager
from MLProject.components.Model_trainer import ModelTrainer
from MLProject import logger
from pathlib import Path


STAGE_NAME = "Model trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__=='__main__':
    try:
        logger.info(f'>>>>>>> Stage {STAGE_NAME} started <<<<<<<<')
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x')
    except Exception  as e:
        logger.exception(e)
        raise e
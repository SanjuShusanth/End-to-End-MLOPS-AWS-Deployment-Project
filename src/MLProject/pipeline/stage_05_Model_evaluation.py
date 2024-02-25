from src.MLProject.config.configuration import ConfigurationManager
from src.MLProject.components.Model_evaluation import ModelEvaluation
from src.MLProject import logger
from pathlib import Path



STAGE_NAME = "Model evaluation Stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evalution_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__=='__main__':
    try:
        logger.info(f'>>>>>>> Stage {STAGE_NAME} started <<<<<<<<')
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x')
    except Exception  as e:
        logger.exception(e)
        raise e
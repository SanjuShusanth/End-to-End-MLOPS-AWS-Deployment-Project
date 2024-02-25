from src.MLProject import logger
from src.MLProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.MLProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.MLProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.MLProject.pipeline.stage_04_model_training import ModelTrainerTrainingPipeline
from src.MLProject.pipeline.stage_05_Model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_transformation = DataTransformationTrainingPipeline() 
    data_transformation.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model trainer training Stage"

try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME} started <<<<<<<<')
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model evaluation Stage"

try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME} started <<<<<<<<')
    model_evalution = ModelEvaluationTrainingPipeline()
    model_evalution.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x')
except Exception as e:
    logger.exception(e)
    raise e

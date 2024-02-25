from src.MLProject.config.configuration import ConfigurationManager
from src.MLProject.components.data_transformation import DataTransformation
from src.MLProject import logger



STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.get_data_transformation_object()
        data_transformation.initiate_data_transformation() 
       
    
if __name__=='__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<<")
        datatransformation = DataTransformationTrainingPipeline()
        datatransformation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx========")

    except Exception as e:
        logger.exception(e)
        raise e
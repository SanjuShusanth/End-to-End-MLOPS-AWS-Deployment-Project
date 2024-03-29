from src.MLProject.config.configuration import ConfigurationManager
from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject import logger



STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


    
if __name__=='__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<<")
        dataingestion = DataIngestionTrainingPipeline()
        dataingestion.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx========")

    except Exception as e:
        logger.exception(e)
        raise e
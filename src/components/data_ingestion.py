import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Insert custom modules directory to the system path
sys.path.insert(0, 'src')

# Import custom logging and exception handling modules
from src.logger import logging
from src.exception import CustomException

# Initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

# Create data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts')

        try:
            logging.info('Reading the dataset')
            df = pd.read_csv(os.path.join('notebooks\data', 'heart_disease_dataset.csv'))
            logging.info('Dataset read as pandas dataframe')

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Performing train-test split')
            train_set, test_set = train_test_split(df, test_size=0.30,random_state=42)

            # Save the train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Error occurred in data ingestion')
            raise CustomException(e, sys)

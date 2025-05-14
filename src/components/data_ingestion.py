# Import standard libraries
import os  # For file path and directory handling
import sys  # To access system-specific parameters and functions

# Import custom exception and logger from the project
from src.exception import customException  # Custom exception handler for better error reporting
from src.logger import logging  # Logger to track the execution flow

# Import pandas for data handling
import pandas as pd

# Import tools for splitting data and defining configurations
from sklearn.model_selection import train_test_split  # To split dataset into training and testing sets
from dataclasses import dataclass  # To simplify class-based configuration management

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


# Configuration class using dataclass to store file paths
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path to store training data
    test_data_path: str = os.path.join('artifacts', 'test.csv')  # Path to store test data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  # Path to store raw data

# Main class for handling data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()  # Initialize configuration object

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log entry into ingestion method
        try:
            # Read raw dataset from CSV file
            df = pd.read_csv(r"E:\Projects\mlproject\notebook\data\stud.csv")
            logging.info('Read the dataset as dataframe')  # Log successful reading of data

            # Create directory for storing artifact files if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log start of train-test split

            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log successful completion

            # Return paths to train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # Raise custom exception with system info in case of error
            raise customException(e, sys)

# If this script is run directly, initiate the data ingestion process
if __name__ =="__main__":
    obj = DataIngestion()  # Create object of DataIngestion
    train_data, test_data =  obj.initiate_data_ingestion()  # Call the method to ingest data

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
           
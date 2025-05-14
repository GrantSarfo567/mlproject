import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import customException  # Custom error handling
from src.logger import logging  # Logging system
import os

from src.utils import save_object

# Configuration class to hold the file path for saving the preprocessor object
@dataclass
class DataTransformationConfig: 
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        # Create instance of config to access the preprocessor path
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function creates and returns a preprocessing object for transforming data.
        It handles both numerical and categorical data pipelines.
        '''
        try: 
            # List of numerical columns to scale
            numerical_columns = ['writing_score', 'reading_score']
            
            # List of categorical columns to encode
            categorical_columns = [
                "gender", 
                "race_ethnicity",
                "parental_level_of_education",  # Comma fixed
                "lunch",
                "test_preparation_course",
            ]
            
            # Numerical pipeline: imputes missing values using median, then scales
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
                
            # Categorical pipeline: imputes missing with most frequent, encodes, then scales
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one-hot-encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # with_mean=False is needed when working with sparse data
                ] 
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info("Categorical Columns Encoding Completed")
            
            # Combine numerical and categorical pipelines into a column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            # Raise a custom exception if something goes wrong
            raise customException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        Applies the transformation object to the training and testing datasets.
        Returns transformed arrays and the saved preprocessor path.
        '''
        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing Object")
            
            # Get the preprocessor pipeline
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            numerical_column = ["writing_score", "reading_score"]
            
            # Separate features and target for training set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # Separate features and target for testing set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes.")
            
            # Apply transformations to training and testing features
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # Concatenate transformed features with target values
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved Preprocessing object.")
            
            # Save the preprocessor object for later use in prediction/inference
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise customException(e, sys)

        

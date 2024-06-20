import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the project directory to the system path
sys.path.insert(0, 'src')

# Import custom modules for logging, exception handling, and utility functions
from src.logger import logging
from src.exception import CustomException
from src.utils import save_objects

# DATA TRANSFORMATION CONFIGURATION
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# DATA TRANSFORMATION CLASS
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated")

            # Define categorical and numerical columns
            categorical_cols = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes',
                                'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']
            numerical_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours',
                              'Stress Level', 'Blood Sugar']

            

            logging.info("Pipelining initiated")
            
            categories = [
                ['Male', 'Female'],  # Gender
                ['Current','Former','Never'],  # Smoking
                ['Heavy', 'Moderate'],  # Alcohol Intake
                ['Yes', 'No'],  # Family History
                ['Yes', 'No'],  # Diabetes
                ['Yes', 'No'],  # Obesity
                ['Yes', 'No'],  # Exercise Induced Angina
                ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']  # Chest Pain Type
            ]

            # Define numerical pipeline
            numerical_pipeline=Pipeline(steps=
                             [('imputer',SimpleImputer(strategy='mean')),
                              ('scaler',StandardScaler())
                             ])



            # Define categorical pipeline
            categorical_pipeline=Pipeline(steps=
                              [('imputer',SimpleImputer(strategy='most_frequent')),
                               ('encoder',OrdinalEncoder(categories=categories)),
                               ('scaler',StandardScaler())])
            # Combine numerical and categorical pipelines
            preprocessor=ColumnTransformer([
                               ('numerical_pipeline',numerical_pipeline,numerical_cols),
                               ('categorical_pipeline',categorical_pipeline,categorical_cols)
                                ])

            logging.info('Pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = 'Heart Disease'
            drop_columns = [target_column_name]

            # Split data into features and target
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            save_objects(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Processor pickle is created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)

from dataclasses import dataclass
import os
import sys
import pandas as pd
from pathlib import Path 
sys.path.insert(0, 'src')
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import numpy as np

class PredictPipeline:
    def _init_(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred


        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
@dataclass
class CustomData:
    Age: float
    Gender: str
    Cholesterol: float
    BloodPressure: float
    HeartRate: float
    Smoking: str
    AlcoholIntake: str
    ExerciseHours: float
    FamilyHistory: str
    Diabetes: str
    Obesity: str
    StressLevel: float
    BloodSugar: float
    ExerciseInducedAngina: str
    ChestPainType: str

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age': [self.Age],
                'Gender': [self.Gender],
                'Cholesterol': [self.Cholesterol],
                'Blood Pressure': [self.BloodPressure],
                'Heart Rate': [self.HeartRate],
                'Smoking': [self.Smoking],
                'Alcohol Intake': [self.AlcoholIntake],
                'Exercise Hours': [self.ExerciseHours],
                'Family History': [self.FamilyHistory],
                'Diabetes': [self.Diabetes],
                'Obesity': [self.Obesity],
                'Stress Level': [self.StressLevel],
                'Blood Sugar': [self.BloodSugar],
                'Exercise Induced Angina': [self.ExerciseInducedAngina],
                'Chest Pain Type': [self.ChestPainType],
            }
            df = pd.DataFrame(custom_data_input_dict)
            
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
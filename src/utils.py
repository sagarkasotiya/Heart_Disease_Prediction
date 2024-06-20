import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name,model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report[model_name] = accuracy
            logging.info(f"Model: {model_name}, Accuracy_score: {accuracy}")
        return report
    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)
    
def save_objects(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error("Error saving object to file")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error("Error loading object from file")
        raise CustomException(e, sys)



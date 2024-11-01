import os,sys,pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info('Error in Utils - Save Object')
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report_result = {}
        for i,j in zip(models.values(),models):
            model = i
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)
            
            score_result = r2_score(y_test,y_pred)
            report_result[j] = score_result

        logging.info('Result Report Ready',report_result)

        return report_result


    except Exception as e:
        logging.info('Error Occured in Utils.evaluate_model during model training')
        raise CustomException(e,sys)
    

def load_pickle_object(file_path):
    try:
        with open(file_path,'rb') as pickle_obj:
            return pickle.load(pickle_obj)

    except Exception as e:
        print('error')
        logging.info('Error in Utils :- load_pickle_objects')
        # raise CustomException(e,sys)
    


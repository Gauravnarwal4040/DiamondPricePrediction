import pandas as pd
import numpy as np
import os,sys

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor

from src.utils import evaluate_model,save_object

@dataclass
class model_pickle_file_path:
    file_path = os.path.join("artifacts","model.pkl")


class model_training:
    
    def train_model(self,train_array,test_array):

        try:

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet(),
            'DecisionTree':DecisionTreeRegressor()}
            print('****')
            X_train_array, y_train_array, X_test_array, y_test_array = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Now we split all train test arrays')
            model_report = evaluate_model(X_train_array,y_train_array,X_test_array,y_test_array,models)
            
            print(model_report)
            logging.info(model_report)

            max_score = max(list(model_report.values()))
            print(max_score)

            best_model_name = [i for i,j in model_report.items() if j == max_score][0]
            print(best_model_name)
            
            logging.info(f'Best Model Found, Model Name :- {best_model_name} , R2_Score :- {max_score}')
            best_model_obj = models[best_model_name]

            save_object(file_path=model_pickle_file_path.file_path,obj=best_model_obj)

            return best_model_obj

        except Exception as e:
            logging.info('Error occur in model training file')
            raise CustomException(e,sys)
    
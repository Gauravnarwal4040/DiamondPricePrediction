import os,sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_pickle_object
from dataclasses import dataclass


@dataclass
class object_class:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
    model_path = os.path.join('artifacts','model.pkl')

    preprocessor_obj  = load_pickle_object(preprocessor_path)
    model_obj = load_pickle_object(model_path)
    logging.info('create preprocessor and model object successfully')


class PredictPipeline:
    def __init__(self):
        self.obj_class = object_class()

    def predict(self,data):
        try:
            preprocessor_obj = self.obj_class.preprocessor_obj
            preprocess_transform = preprocessor_obj.transform(data)

            model_obj = self.obj_class.model_obj
            pred_data = model_obj.predict(preprocess_transform)
            return pred_data
        
        except Exception as e:
            logging.info('Error in PredictPipeline')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


        
    
    

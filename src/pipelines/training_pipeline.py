import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import model_training


logging.info('All Libraries Import Successfully')




if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    logging.info('DataIngestion Complete')
    
    DataTransform_obj = DataTransform()
    train_transform_data, test_transform_data,preprocessorPKL_path = DataTransform_obj.make_data_transformation(train_data_path,test_data_path)
    logging.info('Transformation of Data Complete')
    print('Transformation of data complete')
    logging.info(train_transform_data)

    model_training_obj = model_training()
    model_training_obj.train_model(train_transform_data,test_transform_data)
    


    





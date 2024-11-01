import os,sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Initialize the data ingestion configuration

@dataclass 
class dataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')



# create data ingestion class
class DataIngestion:
    def __init__(self):
        self.data_ingestion = dataIngestionConfig()
        
    def initiate_data_ingestion(self):
            
        try:

            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Reading DataFrame')


            df.to_csv(self.data_ingestion.raw_data_path)
            train_data,test_data = train_test_split(df,random_state=123,test_size=0.30)
            logging.info('Train_Test_Complete')

            train_data.to_csv(self.data_ingestion.train_data_path)
            test_data.to_csv(self.data_ingestion.test_data_path)

            logging.info('Create train, test.csv file successfully')

            return self.data_ingestion.train_data_path,self.data_ingestion.train_data_path
            
        except Exception as e:
            print('Error in Data Ingestion')
            logging.info('Error Occured in Data Ingestion')


            
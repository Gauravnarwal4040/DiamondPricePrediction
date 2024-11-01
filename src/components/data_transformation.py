import pandas as pd
import numpy as np
import os,sys

from src.exception import CustomException
from src.logger import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer         
from sklearn.preprocessing import StandardScaler      ## For Scaling of our data
from sklearn.impute import SimpleImputer              ## Handle Missing Values
from sklearn.preprocessing import OrdinalEncoder      ## Ordinal Encoding ----> Categorical to Numerical data in a order form

from src.components.data_ingestion import DataIngestion
from dataclasses import dataclass
import pickle
from src.utils import save_object


@dataclass
class DataTranformationconfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransform:
    def __init__(self):
        self.data_ingestion_class = DataIngestion()

    def create_transform_data(self):

        try:
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']


            categorical_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('ordinal',OrdinalEncoder(categories = [cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            numerical_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )


            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',numerical_pipeline,numerical_cols),
                    ('cat_pipeline',categorical_pipeline,categorical_cols)
                ]
            )

            return preprocessor
        



        except Exception as e:
            logging.info('Error in Data Transformtion')
            raise CustomException(e,sys)

    def make_data_transformation(self,train_path,test_path):
            try:
                # Reading Train - Test Data
                train_data = pd.read_csv(train_path)
                test_data = pd.read_csv(test_path)

                logging.info('Read Train - Test File')

                target_column_name = 'price'
                drop_columns = ['id',target_column_name]

                train_data_x = train_data.drop(drop_columns,axis=1)
                train_data_y = train_data[target_column_name]

                test_data_x = test_data.drop(drop_columns,axis=1)
                test_data_y = test_data[target_column_name]
                
                data_preprocessor_obj = self.create_transform_data()
                transform_train_data_x = data_preprocessor_obj.fit_transform(train_data_x)
                transfrom_test_data_x = data_preprocessor_obj.transform(test_data_x)

                logging.info("Applying preprocessing object on training and testing datasets.")

                train_final_transformation_data = np.c_[transform_train_data_x,np.array(train_data_y)]
                test_final_transformation_data = np.c_[transfrom_test_data_x,np.array(test_data_y)]


                preprocessor_obj_path = DataTranformationconfig.preprocessor_obj_path

                save_object(preprocessor_obj_path,data_preprocessor_obj)


                logging.info('Now we create pickle file successfully')

                return train_final_transformation_data,test_final_transformation_data,preprocessor_obj_path




            except Exception as e:
                 print('Error in Data Transformation')
                 logging.info('Error in data_transformation',e)
                 pass

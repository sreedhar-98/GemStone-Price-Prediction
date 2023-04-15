import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object

class predictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_object_path=os.path.join('artifacts','transformer.pkl')
            model_object_path=os.path.join('artifacts','best_model.pkl')

            preprocessor=load_object(preprocessor_object_path)
            model=load_object(model_object_path)

            data_scaled=preprocessor.transform(features)
            pred_val=model.predict(data_scaled)

            return pred_val
        except Exception as e:
            logging.info("Exception occured at predictPipeline class")
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
                 carat:float,
                 table:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat=carat
        self.table=table
        self.cut=cut
        self.color=color
        self.clarity=clarity
    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'table':[self.table],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame created")
            return df
        except Exception as e:
            logging.info("Exception occured in prediction pipeline Custom Data class")
            raise CustomException(e,sys)



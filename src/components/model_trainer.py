import os
import sys
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
from src.utils import train_evaluate_model,save_object

@dataclass
class ModelTrainerConfig:
    trainer_object_path=os.path.join('artifacts','best_model.pkl')

class modelTraining:
    def __init__(self):
        self.model_obj_path=ModelTrainerConfig()
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Model Training initiated")
            models_dict={
                'LinearRegression':LinearRegression(),
                'RidgeCV':RidgeCV(alphas=[0.1,0.5,1,2,5,8]),
                'LassoCV':LassoCV(alphas=[0.1,0.5,1,2,5,8]),
                'ElasticNetCV':ElasticNetCV(alphas=[0.1,0.5,1,2,5,8])
                }
            model_report=train_evaluate_model(models_dict,train_arr,test_arr)
            logging.info("Training is completed and the report is generated.")
            logging.info(model_report)
            print('#'*135)
            print(model_report)
            scores=[]
            for model in model_report.index:
                scores.append((model,model_report.loc[model,('Test Dataset','r2 score')]))
        
            scores=sorted(scores,key=lambda x : x[1],reverse=True)

            print('#'*135)
            print('\n')
            print("Model with highest r2 score is {}({})".format(scores[0][0],scores[0][1]))
            logging.info(f"Model with highest r2 score is {scores[0][0]} ({scores[0][1]})")

            save_object(file_path=self.model_obj_path.trainer_object_path,obj=models_dict[scores[0][0]])


        except Exception as e:
            logging.info("Exception occured at model trainer")
            raise CustomException(e,sys)


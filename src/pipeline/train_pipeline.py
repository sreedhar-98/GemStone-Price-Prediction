from src.components.data_ingestion import DataConfig
from src.components.data_transformation import DataTransformer

dc=DataConfig()
train_path,test_path=dc.initiateIngestion()
dt=DataTransformer()
train_arr,test_arr,_=dt.initiate_data_transformation(train_data_path=train_path,test_data_path=test_path)
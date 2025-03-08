#For basic operations
import pandas as pd
import kagglehub
import logging
import warnings

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")

def read_file():
    '''
    Function to read the file from kaggle 
    and return a pandas dataframe
    
    Returns:
    file: pandas DataFrame
    '''
    
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    logging.info(f"Path to dataset files: {path}")

    file = pd.read_csv(f'{path}/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    #Delete rows with all missing values
    file= file.dropna(how='all')

    # Delete duplicate records
    file.drop_duplicates(inplace=True)
    logging.info("File read successfully")
    return file   
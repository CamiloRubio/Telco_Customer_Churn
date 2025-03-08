import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import logging

def new_features(X_train, X_test, X_train_pre):
    '''
    This function creates new features based on the existing ones.
    
    Parameters:
    X_train: DataFrame with the training data
    X_test: DataFrame with the test data
    X_train_pre: DataFrame with the preprocessed training data

    Returns:
    X_train: DataFrame with the training data and the new features
    X_test: DataFrame with the test data and the new features
    '''

    # Feature interaction between Payment and Contract
    X_train['PaymentContract'] = X_train['PaymentMethod'] * X_train['Contract']
    X_test['PaymentContract'] = X_test['PaymentMethod'] * X_test['Contract']
    logging.info("Feature interaction PaymentContract created")
    
    # Feature interaction - Charges_Tenure
    X_train['Charges_Tenure'] = X_train['MonthlyCharges']/X_train['tenure']
    X_test['Charges_Tenure'] = X_test['MonthlyCharges']/X_test['tenure']
    logging.info("Feature interaction Charges_Tenure created")

    # New feature named Family. 1 in case X_train['Partner'] and  X_train['Dependents'] are both 1, 0 else
    X_train['Family'] = ((X_train['Partner'] == 1) & (X_train['Dependents'] == 1)).astype(int)
    X_test['Family'] = ((X_test['Partner'] == 1) & (X_test['Dependents'] == 1)).astype(int)
    logging.info("Feature Family created")

    # Client_segment feature that segments the clients in 4 categories: New, Medium, Loyal and TopLoyal
    mean_tenure = X_train_pre['tenure'].mean()
    std_tenure = X_train_pre['tenure'].std()

    umbral_12_scaled = (12 - mean_tenure) / std_tenure
    umbral_24_scaled = (24 - mean_tenure) / std_tenure

    def segment_client_scaled(row):
        if row['tenure'] < umbral_12_scaled:
            return 'New'
        elif row['tenure'] < umbral_24_scaled:
            return 'Medium'
        else:
            return 'Loyal' if row['MonthlyCharges'] < 0 else 'TopLoyal'

    X_train['Client_Segment'] = X_train.apply(segment_client_scaled, axis=1)
    X_test['Client_Segment'] = X_test.apply(segment_client_scaled, axis=1)

    # Definir el orden de las categorías
    ordered_categories = [['New', 'Medium', 'Loyal', 'TopLoyal']]

    # Inicializar y ajustar el encoder
    encoder = OrdinalEncoder(categories=ordered_categories)
    X_train['Client_Segment'] = encoder.fit_transform(X_train[['Client_Segment']])
    X_test['Client_Segment'] = encoder.transform(X_test[['Client_Segment']])

    logging.info("Feature Client_Segment created")

    return X_train, X_test

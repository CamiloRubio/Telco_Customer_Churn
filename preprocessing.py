#For basic operations
import sys, os
import pandas as pd
import numpy as np
import logging

# For data import
import kagglehub
from read_dataset import read_file

#For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For data preprocessing and splitting
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def exploratory_basics(df):
    '''
    This function delivers a summary of the dataset, including the number 
    of rows and columns, column names, data types, memory usage, 
    number of missing values, number of unique values, summary statistics, and data distribution.

    Parameters:
    df: pandas DataFrame
    '''
    print("Exploratory Data Analysis")
    print("Number of rows: ", df.shape[0])
    print("Number of columns: ", df.shape[1])
    print("Summary statistics: ", df.describe())
    print("Column names: ", df.columns)
    print("Info. Features/Datatypes/Mem. usage: ", df.info(memory_usage='deep'))
    print("Number of missing values: ", df.isnull().sum().sum())
    print("Current level of churn: ", df['Churn'].value_counts(normalize=True))
    

def preprocessing_EDA_and_charts(df, plot_size):
    '''
    This function preprocesses the dataset and creates, shows and saves EDA plots for the dataset.

    Parameters:
    df: pandas DataFrame
    plot_size: int

    Returns:
    df: pandas DataFrame
    categorical_features: list
    numericas_features: list
    '''
    # Conversion of Total Charges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    logging.info("Total Charges converted to numeric")

    # Division between numerical and categorical columns

    # Make categorical features and join df['SeniorCitizen'] to the list:
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    categorical_features = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    numericas_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    logging.info("Columns divided into numerical and categorical")

    print("Unique values in categorical columns: ")
    for col in categorical_features:
        print(col, df[col].unique())
        print("-"*50)
    print("Data distribution: ")
    for col in categorical_features:
        print(df[col].value_counts())
        print("-"*50)

    # Charge null numerical values with the median
    for col in numericas_features:
        df[col] = df[col].fillna(df[col].median())
    logging.info("Numerical columns with null values filled with median")
    
    # Drop  records with null categorical values
    df = df.dropna()
    logging.info("Records with null categorical values dropped")

    # Drop columns with a single category
    df = df[[col for col in df.columns if df[col].nunique() > 1]]
    logging.info("Columns with a single category dropped")

    # Create a directory to save the plots if it does not exist
    save_dir = "charts_pretraining"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sns.set_palette("Blues")
    
    # 1. PAIRPLOT
    # Select only numerical columns
    df_scatter = df.select_dtypes(include=[np.number])
    # Delete columns with null values and those with a single category
    df_scatter = df_scatter.dropna(axis=1)
    df_scatter = df_scatter[[col for col in df_scatter.columns if df_scatter[col].nunique() > 1]]
    n_cols = len(df_scatter.columns)
    # Calculate height of each ceil
    height = plot_size / n_cols
    # Create the pairplot with density plots on the diagonal
    grid = sns.pairplot(df_scatter, diag_kind='kde', diag_kws={'fill': True}, height=height, palette='Blues')
    plt.suptitle('Scatter and Density Plot', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pairplot.png"))
    plt.suptitle('Scatter and Density Plot', y=1.02)
    plt.show()
    plt.close()

    # 2. COUNTPLOT OF CHURN
    plt.figure(figsize=(7,7))
    sns.countplot(x='Churn', data=df, palette= 'Blues')
    plt.title('Churn count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "countplot_churn.png"))
    plt.show()
    plt.close()


    # 3. CHURN PER GENDER
    plt.figure(figsize=(7,7))
    sns.countplot(x='gender', hue='Churn', data=df, palette= 'Blues')
    plt.title('Churn per Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "countplot_churn_gender.png"))
    plt.show()
    plt.close()

    # 4. CHURN PER PAYMENT METHOD
    plt.figure(figsize=(7,7))
    sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette= 'Blues')
    plt.xticks(rotation=45)
    plt.title('Churn per Payment Method')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "countplot_churn_payment.png"))
    plt.show()
    plt.close()

    # 5. DISTRIBUTION OF PAYMENT METHOD
    plt.figure(figsize=(7,7))
    # Count the number of instances of each target class
    paym_met_count = df['PaymentMethod'].value_counts()
    colors = plt.cm.Blues(np.linspace(0.5, 0.8, len(paym_met_count)))
    # Create a pie chart with the target class distribution
    plt.pie(
    paym_met_count, 
    labels=df['PaymentMethod'].unique(), 
    autopct='%1.1f%%',
    colors=colors,
    textprops={'color': 'black', 'backgroundcolor': 'white'}
    )
    plt.title('Gráfico de Sectores - Distribución de Especies') 
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pie_payment_method.png"))
    plt.show()
    plt.close()

    # 6. CHURN PER CONTRACT
    plt.figure(figsize=(7,7))
    sns.countplot(x='Contract', hue='Churn', data=df, palette= 'Blues')
    plt.xticks(rotation=45)
    plt.title('Churn per Contract')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "countplot_churn_contract.png"))
    plt.show()
    plt.close()

    # 7. HEATMAP OF CORRELATION FOR NUMERICAL FEATURES
    df_heat = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(7, 7))
    sns.heatmap(df_heat.corr(), annot=True, fmt = ".2f", cmap='Blues')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap_correlation.png"))
    plt.show()

    # 8. BOX PLOT OF CHURN PER TENURE
    plt.figure(figsize=(7, 7))
    sns.boxplot(x='Churn', y='tenure', data=df, palette= 'Blues')
    plt.title('Boxplot of Churn per Tenure')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "boxplot_churn_tenure.png"))
    plt.show()
    plt.close()

    # 9. BOX PLOT OF CHURN PER MONTHLY CHARGES
    plt.figure(figsize=(7, 7))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette= 'Blues')
    plt.title('Boxplot of Churn per Monthly Charges')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "boxplot_churn_monthly_charges.png"))
    plt.show()
    plt.close()

    #10. TOTAL CHARGES VS. MONTHLY CHARGES PER CONTRACT
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Contract', data=df, palette= 'Blues')
    plt.title('Total Charges vs. Monthly Charges per Contract')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatterplot_total_monthly_charges_contract.png"))
    plt.show()
    plt.close()

    return df, categorical_features, numericas_features

def ultimate_preprocessing(df, numerical_features, categorical_features):
    '''
    This function performs additional preprocessing tasks on the dataset.

    Parameters:
    df: pandas DataFrame
    numerical_features: list
    categorical_features: list

    Returns:
    X: pandas DataFrame
    y: pandas DataFrame
    X_train: pandas DataFrame
    X_test: pandas DataFrame
    y_train: pandas DataFrame
    y_test: pandas DataFrame
    X_res: pandas DataFrame
    y_res: pandas DataFrame
    X_train_pre: pandas DataFrame
    '''
    # Drop Customer Id - unnecessary for the model
    df = df.drop(columns=['customerID'])
    logging.info("CustomerID column dropped")

    # Split dataset after scaling and encoding
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    logging.info("Dataset splitted into X and y")

    # Apply train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_pre = X_train.copy()
    logging.info("Dataset splitted into train and test")

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    logging.info("Numerical columns scaled")

    # Apply LabelEncoder:
    le = LabelEncoder()
    categorical_features = [col for col in categorical_features if col != 'Churn' and col != 'customerID']
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        le_dict[col] = le

    for col in categorical_features:
        X_test[col] = le_dict[col].transform(X_test[col])

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    logging.info("Categorical dependent column encoded")


    # Apply SMOTE to partially balance the dataset
    smote = SMOTE(sampling_strategy=0.66666666667, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logging.info("Dataset partially balanced with SMOTE")

    return X, y, X_train, X_test, y_train, y_test, X_res, y_res, X_train_pre



    




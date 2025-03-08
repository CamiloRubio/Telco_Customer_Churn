# For basic operations
import time
import pandas as pd
import numpy as np
import logging

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For modelling, optimization and evaluation
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, 
                             confusion_matrix, roc_auc_score, roc_curve, brier_score_loss, auc)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import optuna
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

# For Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, AUC, BinaryAccuracy

# For explaining models
import shap
import random

# --- Function to build a neural network model with Keras ---
def build_keras_model(n_hidden, n_neurons, dropout_rate, learning_rate, input_dim):
    '''
    This function builds a neural network model using Keras.
    
    Parameters:
    n_hidden: int
    n_neurons: int
    dropout_rate: float
    learning_rate: float
    input_dim: int
    
    Returns:
    model: Keras Sequential model
    '''
    model = Sequential()
    # Input layer and first hidden layer
    model.add(Dense(n_neurons, activation='relu', input_dim=input_dim))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    # Additional hidden layers (if required)
    for _ in range(n_hidden - 1):
        model.add(Dense(n_neurons, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    return model

# --- General training and evaluation function ---
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, is_keras=False, epochs=10):
    '''
    This function trains a model and evaluates it using the test set.
    
    Parameters:
    model: scikit-learn or Keras model
    model_name: str
    X_train: numpy array
    y_train: numpy array
    X_test: numpy array
    y_test: numpy array
    is_keras: bool
    epochs: int
    '''
    # Measure training time
    start_time = time.time()
    if is_keras:
        model.fit(X_train, y_train, epochs=epochs, verbose=0)
    else:
        model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Obtaining predictions and probabilities
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For Keras models, we use predict and flatten the result
        y_pred_prob = model.predict(X_test).ravel()
    # Fixed threshold of 0.5 for classification
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Metrics calculation
    metrics = {}
    metrics['Model'] = model_name
    metrics['Accuracy'] = accuracy_score(y_test, y_pred)
    metrics['Recall'] = recall_score(y_test, y_pred)
    metrics['Precision'] = precision_score(y_test, y_pred)
    metrics['F1'] = f1_score(y_test, y_pred)
    metrics['ROC_AUC'] = roc_auc_score(y_test, y_pred_prob)
    metrics['Brier'] = brier_score_loss(y_test, y_pred_prob)
    metrics['Time'] = training_time
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predictions')
    plt.ylabel('Real')
    plt.show()
    
    # Probability histogram
    plt.figure(figsize=(10,6))
    sns.histplot(y_pred_prob[y_test==0], color='skyblue', kde=True, label='Negatives')
    sns.histplot(y_pred_prob[y_test==1], color='red', kde=True, label='Positives')
    plt.title(f'Probabilities Histogram - {model_name}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    disp = CalibrationDisplay(prob_true, prob_pred, y_pred_prob)
    disp.plot()
    plt.title(f'Calibration Curve - {model_name}')
    plt.show()
    
    return metrics

# --- Function to select the best model based on the metrics ---
def select_best_model(metrics_df):
    """
    F1, ROC_AUC and Brier metrics are compared:
    - F1 and ROC_AUC: higher is better.
    - Brier: lower is better.
    In case of a tie (no model wins in at least 2 metrics),
    the one with the lowest training 'Time' is chosen.

    Parameters:
    metrics_df: DataFrame with metrics for each model

    Returns:
    best: str, the name of the best model
    wins: dict, number of wins for each
    """
    wins = {model: 0 for model in metrics_df['Model']}
    
    # It is assumed that there are 3 rows (for 3 models)
    models = metrics_df.index.tolist()
    for metric in ['F1', 'ROC_AUC']:
        best_model = metrics_df[metric].idxmax()
        wins[metrics_df.loc[best_model, 'Model']] += 1

    # For Brier (smaller is better)
    best_brier = metrics_df['Brier'].idxmin()
    wins[metrics_df.loc[best_brier, 'Model']] += 1

    # Select the model with the most "wins"
    best = max(wins, key=wins.get)
    # If there is a tie, break the tie with the training time
    if list(wins.values()).count(wins[best]) > 1:
        best_time = metrics_df.loc[metrics_df['Model'] == best, 'Time'].values[0]
        for idx in metrics_df.index:
            if metrics_df.loc[idx, 'Time'] < best_time:
                best = metrics_df.loc[idx, 'Model']
                best_time = metrics_df.loc[idx, 'Time']
    return best, wins

# --- Generator function for optimization with Optuna ---
def get_objective(model_name, X_train, y_train, X_test, y_test):
    '''
    This function generates an objective function for Optuna optimization.
    
    Parameters:
    model_name: str
    X_train: numpy array
    y_train: numpy array
    X_test: numpy array
    y_test: numpy array
    
    Returns:
    objective: function
    '''
    def objective(trial):
        '''
        This function defines the objective for the Optuna optimization.
        
        Parameters:
        trial: Optuna trial object
        
        Returns:
        roc_auc: float
        '''
        if model_name == 'Random Forest':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            clf = RandomForestClassifier(n_estimators=n_estimators, 
                                         max_depth=max_depth, 
                                         min_samples_split=min_samples_split)
            clf.fit(X_train, y_train)
            y_pred_prob = clf.predict_proba(X_test)[:, 1]
        elif model_name == 'XGBoost':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 2, 20)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            clf = xgb.XGBClassifier(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    learning_rate=learning_rate,
                                    use_label_encoder=False, 
                                    eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred_prob = clf.predict_proba(X_test)[:, 1]
        elif model_name == 'Keras NN':
            n_hidden = trial.suggest_int('n_hidden', 1, 3)
            n_neurons = trial.suggest_int('n_neurons', 16, 128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            epochs = trial.suggest_int('epochs', 10, 50)
            input_dim = X_train.shape[1]
            model = build_keras_model(n_hidden, n_neurons, dropout_rate, learning_rate, input_dim)
            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            y_pred_prob = model.predict(X_test).ravel()
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        return roc_auc
    return objective


def run_cross_validation(best_model, study, X_train, y_train):
    """
    Runs cross-validation (CV) for the selected model.

    Parameters:
    - best_model: str, can be 'Keras NN', 'XGBoost' or 'Random Forest'.
    - study: object containing study.best_params with the optimal hyperparameters.
    - X_train: input data.
    - y_train: labels.

    Returns:
    - scores: array of scores obtained in each fold.
    """
    logging.info(f"Running cross-validation for {best_model}...")
    if best_model == 'Keras NN':
        # Hyperparameters for the neural network
        n_hidden = study.best_params.get('n_hidden')
        n_neurons = study.best_params.get('n_neurons')
        dropout_rate = study.best_params.get('dropout_rate')
        learning_rate = study.best_params.get('learning_rate')
        epochs = study.best_params.get('epochs')
        input_dim = X_train.shape[1]

        # Function that builds the Keras model (make sure you have build_keras_model implemented)
        def build_model():
            return build_keras_model(n_hidden, n_neurons, dropout_rate, learning_rate, input_dim)
        model = KerasClassifier(build_fn=build_model, epochs=epochs, verbose=0)

    elif best_model == 'XGBoost':
        # Hyperparameters for XGBoost
        n_estimators = study.best_params.get('n_estimators')
        max_depth = study.best_params.get('max_depth')
        learning_rate = study.best_params.get('learning_rate')
        subsample = study.best_params.get('subsample', 1.0)
        colsample_bytree = study.best_params.get('colsample_bytree', 1.0)
        # Build XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate, 
            subsample=subsample, 
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    elif best_model == 'Random Forest':
        # Import RandomForest
        from sklearn.ensemble import RandomForestClassifier
        # Hyperparameters for Random Forest
        n_estimators = study.best_params.get('n_estimators')
        max_depth = study.best_params.get('max_depth')
        random_state = study.best_params.get('random_state', 42)
        # Build Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
    else:
        raise ValueError(f"Modelo no soportado: {best_model}")

    # Run cross-validation using 5 folds and ROC AUC metric
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    logging.info(f"Cross validation process ended for {best_model}.")
    print(f"ROC AUC por fold ({best_model}): {scores}")
    print(f"ROC AUC promedio ({best_model}): {scores.mean():.4f}")
    
    return scores



def plot_feature_importances_or_shap(best_model, best_model_optimized, X_train, X_test):
    """
   Displays feature importances if the best model is Random Forest or XGBoost
    or uses SHAP to explain a Keras neural network.

    Parameters:
    -----------
    best_model : str
    Name of the best model (e.g. 'Random Forest', 'XGBoost', 'Keras NN').
    best_model_optimized : model object
    Instance of the best trained model.
    X_train : pd.DataFrame or pd.Series
    Training data (features).
    X_test : pd.DataFrame or pd.Series
Test data (features).
    """

    # --- Case 1: Random Forest or XGBoost ---
    if best_model in ['Random Forest', 'XGBoost']:
        # We verify that the model has the attribute 'feature_importances_'
        if hasattr(best_model_optimized, 'feature_importances_'):
            feature_importances = best_model_optimized.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': feature_importances
            })
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10,8))
            sns.barplot(
                x='Importance',
                y='Feature',
                data=feature_importance_df,
                color='skyblue'
            )
            plt.title(f'Feature Importances - {best_model}')
            plt.show()
        else:
            print(f"El modelo {best_model} no tiene atributo 'feature_importances_'.")
    
    # --- Case 2: Keras NN ---
    elif best_model == 'Keras NN':
            # We create the explainer and calculate SHAP values
            explainer = shap.Explainer(best_model_optimized, X_train)
            shap_values = explainer(X_test) 
            # Bar plot
            shap.plots.bar(shap_values, max_display= 23)
            # Beeswarm plot
            shap.plots.beeswarm(shap_values, max_display= 23)
    else:
        print(f"El modelo '{best_model}' no está contemplado en esta función.")
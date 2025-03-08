# Telco_Customer_Churn_Prediction_Model

This repository contains a Pipeline for a machine learning model designed to predict whether a telecommunications customer will churn or continue using the service. The model was developed using the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data).

## Problem Definition
The primary objective is to build a predictive model that accurately determines if a telecom customer is likely to abandon the service (churn) or remain loyal. Accurately predicting customer churn enables the company to identify at-risk customers and implement proactive retention strategies—an essential factor for maintaining business profitability.

## Evaluation Metrics
Given the inherent class imbalance (fewer customers churn compared to those who stay), selecting the right evaluation metrics is crucial. The following metrics were considered to assess the performance of the model:

* **Accuracy:**
Represents the overall percentage of correct predictions. However, in imbalanced datasets, this metric can be misleading.

* **Precision:**
Measures the proportion of true positive predictions among all positive predictions. This indicates how accurate the positive (churn) predictions are.

**Recall (Sensitivity):**
Measures the proportion of actual positives that are correctly identified by the model. This is critical for capturing as many potential churners as possible.

* **F1-Score:**
The harmonic mean of precision and recall, providing a balanced view when there is a trade-off between the two.

* **ROC-AUC (Area Under the ROC Curve):**
Evaluates the model’s ability to distinguish between the classes across different threshold values. This metric is especially valuable in the context of imbalanced datasets.

* **Brier Score:**
Measures the accuracy of probabilistic predictions by calculating the mean squared difference between predicted probabilities and the actual outcomes. A lower Brier Score indicates better calibrated predictions.

## Visualizations for Stakeholders
To effectively communicate the results to a non-technical audience, the following visualizations were prepared:

* *Simplified Confusion Matrix:*
A clear representation of the model’s correct and incorrect predictions, highlighting the true positives, false positives, true negatives, and false negatives.

* *Key Metrics Summary:*
A concise presentation of the primary metrics, such as recall, precision, F1 score, and ROC-AUC, that succinctly demonstrates the model’s performance.

* *Feature Importance Graph:*
A bar chart illustrating the most influential features in predicting churn, which provides valuable insights into the factors driving customer attrition.

## Modeling Pipeline
The notebook for the current project follows a comprehensive modeling pipeline that includes the following stages:

* Data Preprocessing:

Cleaning and preparing the dataset
Handling missing values
Encoding categorical variables
Normalizing numerical features
Splitting the data into training and testing sets

* Feature Engineering:

Creating and transforming variables to improve model performance
Utilizing domain knowledge and exploratory data analysis to derive meaningful features

* Model Training and Validation:

Implementing and training various classification algorithms (e.g., Random Forest, XGBoost, Neural Networks)
Cross-validation was applied to verify that the model's metrics were stable.

* Final Evaluation:

Assessing model performance on the test set using the selected evaluation metrics
Analyzing the confusion matrix and comparing results against the predefined business objectives

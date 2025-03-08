import os
import logging
import pickle
import pandas as pd
import numpy as np
import time

from dash import dcc, html
from dash import dash_table
import dash_bootstrap_components as dbc
import dash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from read_dataset import read_file
from preproc_app import exploratory_basics, preprocessing_EDA_and_charts, ultimate_preprocessing
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, brier_score_loss, classification_report
)

# Import graphics functions and new features.
from utils import (
    create_correlation_heatmap,
    create_churn_countplot,
    create_churn_gender_countplot,
    create_contract_churn_plot,
    create_boxplot_monthlycharges,
    create_boxplot_tenure,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_calibration_curve_plot,
    create_shap_summary_plot,
    class_report,
    create_metrics_table,
    churn_info_card
)
from feature_engineering import new_features

logging.basicConfig(level=logging.INFO)

# Load the dataset (adjust the path as appropriate)
df = read_file()

# Separate numerical and categorical variables
df_preprocessed, categorical_features, numerical_features = preprocessing_EDA_and_charts(df, 8)

# Split into training and testing
X, y, X_train, X_test, y_train, y_test, X_res, y_res, X_train_pre = ultimate_preprocessing(df_preprocessed, numerical_features, categorical_features)


# Generate new features using the preprocessed copy (X_train_pre)
# X_train_pre is assumed to contain at least the original required columns: PaymentMethod, Contract, MonthlyCharges, tenure, Partner, Dependents
X_train, X_test = new_features(X_res, X_test, X_train_pre)

# Load the saved model (best_model.pkl)
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Obtenain predictions and probabilities
y_pred_opt_prob = best_model.predict(X_test)
y_pred_opt = (y_pred_opt_prob > 0.5).astype(int)

metrics_optimized = {
    "Modelo": 'Keras Neural Network',
    "Accuracy": accuracy_score(y_test, y_pred_opt),
    "Recall": recall_score(y_test, y_pred_opt),
    "Precision": precision_score(y_test, y_pred_opt),
    "F1": f1_score(y_test, y_pred_opt),
    "ROC_AUC": roc_auc_score(y_test, y_pred_opt_prob),
    "Brier": brier_score_loss(y_test, y_pred_opt_prob)
}
df_metrics_optimized = pd.DataFrame([metrics_optimized])
print("Tabla de métricas del mejor modelo optimizado:")
print(df_metrics_optimized)

from dash import html
import dash_bootstrap_components as dbc


metrics_table = create_metrics_table(df_metrics_optimized)


# Create the feature importance graph (using SHAP) on a sample of X_test
shap_fig = create_shap_summary_plot(best_model, X_test)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout with the new graphics
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Telco Customer Churn Dashboard"), width=12)),

    # Correlation Heatmap
    dbc.Row(dbc.Col(dcc.Graph(figure=create_correlation_heatmap(df_preprocessed)), width=12)),

    dbc.Row([
        dbc.Col(html.P('''
        From the heatmap, you can see that tenure (the length of time a customer has been with the company) has a strong positive correlation with TotalCharges.
        Specifically, the correlation coefficient between these two variables is around 0.825, which is considered a high value.
        This indicates that as a customer’s tenure increases, their total charges also tend to increase.
        The reasoning behind this is fairly intuitive: customers who remain subscribed for a longer period generally accumulate more charges over time.
        In other words, the longer someone stays with the service, the more they pay in total, hence the strong correlation.'''), width=12)
    ]),

    # Counting Charts: Churn and Churn by Gender
    dbc.Row([
        dbc.Col(dcc.Graph(figure=create_churn_countplot(df_preprocessed)), width=6),
        dbc.Col(dcc.Graph(figure=create_churn_gender_countplot(df_preprocessed)), width=6)
    ]),

    dbc.Row([
    dbc.Col([
        html.P('''
            The proportion of male and female dropouts and non-quitters in the base is quite similar. However, in aggregate terms there is an imbalance in the data, as approximately 70% of clients do not drop out while 
            30% have dropped out. 
            Since we do not know how costly it might be to intervene to retain a customer, we will assume that the cost of inaction is probably higher than the cost of applying retention measures to a customer who is not 
            at risk of churn. Therefore, we prioritize recall and apply the SMOTE technique to achieve a balance of 60% for class 0 and 40% for class 1, this because usually the cost of loosing a client is higher than the retain cost.
        '''),
        html.Br(),
        html.P('''
            This adjustment significantly increased recall while precision decreased only slightly. However, recall was not used as a criterion for selecting the best model; because the data remains unbalanced,
            we considered the F1 Score, Brier Score, and especially the ROC-AUC as the primary metrics for selecting the best model, rather than relying solely on accuracy. The ROC-AUC curve served as the key metric 
            when optimizing the model. The reason is that this metric evaluates the true positive rate against the false positive rate across all possible decision thresholds, measures the model's ability to assign higher 
            probabilities to positive cases compared to negative ones, and is insensitive to class imbalance.
        ''')
    ], width=12)
    ]),


    # Interactive chart of churn per contract
    dbc.Row(dbc.Col(dcc.Graph(figure=create_contract_churn_plot(df_preprocessed)), width=12)),

    dbc.Row([
    dbc.Col([
        html.P('''
            A relatively high proportion of customers with a month-to-month contract have dropped out, while the proportion of customers with a two-year contract who have dropped out is much lower.
            This suggests that customers with a month-to-month contract are more likely to churn than those with a longer-term contract. This is an important insight for the company, as it indicates that customers
            with a month-to-month contract may require additional attention or incentives to remain with the service.
        '''),
        html.Br(),
        html.P('''
            Usually when customers pay for a month-to-month subscription, they have more freedom to cancel their service at any time without facing large penalties or long-term commitments. 
            Then lower barrier exists in this case and customers are more likely to churn. Also, they can be more aware of the total monthly cost of the service and may be more sensitive to price changes, because
            they are not locked into a long-term contract where the price is fixed for a certain period of time and you are not constantly reminded of the cost.
        ''')
    ], width=12)
    ]),


   # Interactive Boxplots: Monthly Charges and Tenure
    dbc.Row(dbc.Col(dcc.Graph(figure=create_boxplot_monthlycharges(df_preprocessed)), width=12)),
    dbc.Row(dbc.Col(dcc.Graph(figure=create_boxplot_tenure(df_preprocessed)), width=12)),

    dbc.Row([
        dbc.Col(html.P('''
        Customers who pay higher monthly payments are at higher risk of churn, and customers with longer tenure are less likely to churn.'''), width=12)
    ]),

    # Model evaluation graphs
    dbc.Row([
        dbc.Col(dcc.Graph(figure=create_confusion_matrix_plot(best_model, X_test, y_test)), width=6),
        dbc.Col(dcc.Graph(figure=create_roc_curve_plot(best_model, X_test, y_test)), width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.P('''
            From the confusion matrix, we can see that:
                   '''
            ),
                   html.Ul([
                          html.Li("258 customers were correctly identified as churn (true positives)."),
                            html.Li("851 customers were correctly identified as non-churn (true negatives)."),
                            html.Li("185 non-churn customers were incorrectly flagged as churn (false positives)."),
                            html.Li("115 actual churners were missed by the model (false negatives).")
                     ]),
            html.Br(),
            html.P('''
                   This breakdown highlights both the model’s ability to correctly classify most customers (the true positives and true negatives) 
                   and the specific areas where it may struggle (the false positives and false negatives). In a churn context, false negatives are especially 
                   costly since they represent customers who will leave but were not flagged for potential retention efforts.
                   '''),
            html.Br(),
            html.P('''Turning to the ROC curve, we note that it plots the true positive rate (recall) against the false positive rate at various classification thresholds. 
                   The AUC (Area Under the Curve) of 0.85 indicates that the model generally does a good job distinguishing between churners and non-churners across different probability cutoffs. 
                   A higher AUC signifies better separability; here, a value of 0.85 suggests a strong classification performance. Don't forget it was the metric the best model was optimized for.
                   '''),
            html.Br(),
            html.P('''
                   Overall, while the model correctly identifies most churners, there is room for improvement in reducing false negatives—ensuring 
                   fewer churners slip through undetected—and false positives, so fewer loyal customers are targeted unnecessarily.'''
            ),
            html.Br(),
            html.P('''
            The ROC curve shows that the model has a high true positive rate (recall) and a low false positive rate across all possible decision thresholds. 
            This indicates that the model is able to assign higher probabilities to positive cases compared to negative cases, and is insensitive to class imbalance. 
            The area under the ROC curve (ROC-AUC) is a measure of the model's ability to discriminate between positive and negative cases, with a value of 0.85 indicating good discrimination.
            ''')
        ], width=12)
    ]),

    # Interactive calibration curve
    dbc.Row(dbc.Col(dcc.Graph(figure=create_calibration_curve_plot(best_model, X_test, y_test)), width=12)),

    dbc.Row([
        dbc.Col([
            html.P('''
            This calibration curve compares the mean predicted probability of churn on the x-axis with the actual fraction of positives on the y-axis.
            The diagonal dashed line represents perfect calibration, where predicted probabilities align exactly with observed outcomes.
            '''),
        html.Br(),
        html.P('''
            For most probability ranges, the curve remains reasonably close to the diagonal, indicating that when the model assigns a certain probability of churn, 
            the observed rate of churn is not drastically different.
            In the mid-range (around 0.3–0.5), the curve dips below the diagonal, suggesting the model slightly overestimates the probability of churn in that region. 
            Conversely, near higher predicted probabilities (0.8 and above), the curve rises above the diagonal, hinting at a modest underestimation in those higher bands.
            Despite these deviations, the model’s calibration is decent. The closer the curve is to the diagonal, the more reliable the predicted probabilities are in reflecting the true likelihood of churn.
            '''),
        html.Br(),
        html.P('''
           A well-calibrated model is valuable because its predicted probabilities can be used to make more informed business decisions—such as targeting interventions based on specific probability 
            thresholds—knowing those probabilities meaningfully represent actual risk.
            ''') 
        ], width=12)
    ]),

    # Section for model metrics - Class 1
    dbc.Row(dbc.Col(html.H3("Model Metrics Summary - Class 1"), width=12)),
    dbc.Row(dbc.Col(metrics_table, width=12)),

    # Section for the classification report
    dbc.Row(dbc.Col(html.H3("Full Classification Report"), width=12)),
    dbc.Row(dbc.Col(dcc.Graph(figure=class_report(y_test, y_pred_opt)), width=12)),

    
    dbc.Row([
        dbc.Col([
            html.H4('''
            Model Metrics Summary – Class 1
            '''),
            html.Br(),
            html.Ul([
                html.Li('Accuracy (≈ 78.8%): Out of all customers in the dataset, nearly 79% are classified correctly.'),
                html.Li('Recall (≈ 69.17%): Of the customers who actually churn (class 1), the model captures around 69%. This is crucial for churn scenarios, as missing a churn (false negative) can be costly.'),
                html.Li('Precision (≈ 58.23%): When the model predicts churn, it is correct about 58% of the time. This indicates there are some false positives—non-churners flagged as churn.'),
                html.Li('F1 Score (≈ 63.25%): Balances precision and recall. At around 63%, it suggests there’s a moderate balance between identifying true churners and minimizing false alarms.'),
                html.Li('ROC_AUC (≈ 0.85): Shows that, across various thresholds, the model can generally separate churners from non-churners quite well. A score closer to 1.0 indicates better separability.'),
                html.Li('Brier (≈ 0.15): Measures how well-calibrated the predicted probabilities are. A lower score is better, and 0.15 suggests moderate calibration.')
            ]),
            html.Br(),
            html.H4('''
            Full Classification Report
            '''),
            html.Br(),
            html.Ul([
                html.Li('Class 0 (No Churn) has higher precision, recall, and F1, indicating the model is more confident and accurate in identifying non-churners.'),
                html.Li('Class 1 (Churn) shows lower precision and recall. This means the model struggles more with predicting churners correctly and can mislabel some non-churners as churn.'),
                html.Li('The support values reflect the number of samples for each class, which can impact the balance of precision and recall.'),
                html.Li('The weighted average and macro average metrics help interpret overall performance when classes are imbalanced.'),
            ]),
            html.Br(),
            html.P('''
            In summary, the model is fairly strong at detecting non-churners, but there’s room for improvement in pinpointing and minimizing missed churners (false negatives) and reducing the number 
            of false alarms (false positives). Improving recall and precision for the churn class can lead to more effective customer-retention strategies.
            ''')
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col(churn_info_card(df_preprocessed, y_pred_opt), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.P('''
            The model predicts that 443 customers will churn, which represents slightly more tha 30% of the total customer base. This is a significant number of customers at risk of leaving the service.
            ''')
        ], width=12)
    ]), 

    # Interactive feature importance chart (SHAP)
    dbc.Row(dbc.Col(dcc.Graph(figure=create_shap_summary_plot(best_model, X_test)), width=12)),

    dbc.Row([
        dbc.Col([
            html.P('''
                   This bar chart ranks the features based on their average absolute contribution to the model’s prediction of churn, as measured by SHAP values. 
                   The longer the bar, the more influential that feature is in pushing predictions toward “churn” or “non-churn.”
                   '''),
            html.Br(),
            html.P('''
            According to the findings in the Feature Importances section, the hypothesis formed during the exploratory data analysis is confirmed: the type of contract a customer has influences their likelihood of churning,
            as does the amount paid per month and their level of seniority. The impact that tenure generates, reflects that customers with longer tenure tend to be less likely to churn. This makes sense because loyal customers are often more invested 
            in the service or face higher switching costs.
                   '''),
            html.Br(),
            html.P('''
            Also, the Tech Support feature Indicates whether customers have technical support. Lack of support can increase dissatisfaction, so it’s not surprising that it influences churn predictions.
            A particularly notable engineered feature is Client_Segment, which groups customers into categories based on their tenure and monthly charges:
            '''),
            html.Br(),
            html.Ul([
                html.Li('New: Tenure < 12 months'),
                html.Li('Medium: Tenure between 12 and 24 months'),
                html.Li('Loyal: Tenure > 24 months and monthly charges below the mean'),
                html.Li('Top Loyal: Tenure > 24 months and monthly charges above the mean')
            ]),
            html.Br(),
            html.P('''
                   This feature combines the duration of a customer’s relationship with the company and their billing level to capture not just how long they’ve stayed, 
                   but also how much they typically pay. By doing so, it can reveal nuanced patterns—such as whether long-standing customers with higher bills behave differently 
                   from newer or medium-tenure customers.
                   '''),
            html.Br(),

            html.P('''
                   In the SHAP chart, the presence (and ranking) of Client_Segment indicates how valuable this custom grouping is for the model. 
                   If it ranks highly, it suggests that combining tenure and monthly charges into these distinct segments meaningfully helps the model differentiate between those more or less likely to churn.
                   '''),
            
             html.Br(),
            html.P('''                    

            A good way to retain customers might be by offering annual plans at a discounted price compared to the original rate for the first year, and then charging the normal price from the second year onward; 
                   surely, the customer won't leave even after being charged for the second payment. In the future, it would also be worthwhile to specifically identify the cost of retaining a customer who never
                    really intended to leave, thereby providing a clearer perspective on the preference between metrics such as precision or recall.
''')    
        ], width=12)
        ])

], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
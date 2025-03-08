import os
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
from dash import dcc, html
import dash_bootstrap_components as dbc
import shap
import plotly.graph_objects as go
from sklearn.metrics import classification_report

def create_correlation_heatmap(df, plot_size=7):
    """
    Creates an interactive heatmap of the correlation of numerical variables using Plotly.
    Only numerical columns are selected and the correlation matrix is calculated.

    Parameters:
    df: pandas DataFrame
    plot_size: int
    """
    df_heat = df.select_dtypes(include=[np.number])
    corr = df_heat.corr()
    # Convert correlation matrix to long format for plotly
    corr_long = corr.reset_index().melt(id_vars='index')
    corr_long.columns = ['Variable1', 'Variable2', 'Correlation']
    
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=px.colors.sequential.Blues,
        title="Correlation Heatmap"
    )    
    fig.update_layout(width=plot_size*100, height=plot_size*100)
    return fig

def create_churn_countplot(df):
    """
    Generates an interactive bar chart to count the values of 'Churn'.

    Parameters:
    df: pandas DataFrame
    """
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    fig = px.bar(
        churn_counts,
        x='Churn',
        y='Count',
        color='Churn',
        title="Churn Count",
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    return fig

def create_churn_gender_countplot(df):
    """
    Generates an interactive bar chart to count the values of 'Churn' per gender.

    Parameters:
    df: pandas DataFrame
    """
    fig = px.histogram(
        df,
        x='gender',
        color='Churn',
        barmode='group',
        title="Churn per Gender",
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    return fig

def create_contract_churn_plot(df):
    """
    Generates an interactive bar chart to show the churn per contract.

    Parameters:
    df: pandas DataFrame
    """
    fig = px.histogram(
        df,
        x='Contract',
        color='Churn',
        barmode='group',
        title="Churn per Contract",
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    return fig

def create_boxplot_monthlycharges(df):
    """
    Generates an interactive boxplot of Churn per Monthly Charges.

    Parameters:
    df: pandas DataFrame    
    """
    fig = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        title="Boxplot of Churn per Monthly Charges",
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    return fig

def create_boxplot_tenure(df):
    """
    Generates an interactive boxplot of Churn per Tenure.

    Parameters:
    df: pandas
    """
    fig = px.box(
        df,
        x='Churn',
        y='tenure',
        color='Churn',
        title="Boxplot of Churn per Tenure",
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    return fig

def create_confusion_matrix_plot(model, X_test, y_test):
    """
    Computes the confusion matrix and visualizes it interactively.
    It is assumed that model.predict returns probabilities and it is converted to 0/1.

    Parameters:
    model: trained model
    X_test: pandas DataFrame
    y_test: pandas DataFrame
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm, 
        text_auto=True, 
        color_continuous_scale='Blues', 
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Churn', 'Churn'],
        y=['No Churn', 'Churn']
    )
    fig.update_layout(title="Confusion Matrix")
    return fig

def create_roc_curve_plot(model, X_test, y_test):
    """
    Generates an interactive ROC curve along with the AUC value.

    Parameters:
    model: trained model
    X_test: pandas DataFrame
    y_test: pandas DataFrame
    """
    y_probs = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig = px.area(
        x=fpr, 
        y=tpr, 
        title=f'ROC Curve (AUC={roc_auc:.2f})',
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        width=600, 
        height=400
    )
    fig.add_shape(
        type='line', 
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    return fig

def create_calibration_curve_plot(model, X_test, y_test):
    """
    Generates an interactive calibration curve.
    The calibration is computed using sklearn's calibration_curve and plotted with Plotly.

    Parameters:
    model: trained model
    X_test: pandas DataFrame
    y_test: pandas DataFrame
    """
    y_probs = model.predict(X_test)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    fig = px.line(
        x=prob_pred, 
        y=prob_true, 
        markers=True,
        title="Calibration Curve",
        labels={'x': 'Mean Predicted Probability', 'y': 'Fraction of Positives'}
    )
    # Ideal line
    fig.add_shape(
        type='line', 
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    return fig

def create_shap_summary_plot(model, X_sample):
    """"
    Generates a SHAP summary plot using the KernelExplainer.
    It is assumed that model.predict returns probabilities.

    Parameters:
    model: trained model
    X_sample
    """
    # Use only a subset of the data to speed up computation
    X_sample = X_sample.iloc[:100]
    explainer = shap.KernelExplainer(model.predict, X_sample, link="logit")
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    # If it is a binary classifier, shap_values is a list: select the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_importance = np.abs(shap_values).mean(axis=0).flatten()
    
    df_shap = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': shap_importance
    })
    df_shap = df_shap.sort_values('importance', ascending=True)
    
    fig = px.bar(
        df_shap, 
        x='importance', 
        y='feature', 
        orientation='h', 
        title="Feature Importance (SHAP)",
        color='importance',
        color_continuous_scale=px.colors.sequential.Blues
    )
    return fig

def class_report(y_true, y_pred):
    """"
    Generates an interactive table with the classification report.

    Parameters:
    y_true: pandas DataFrame
    y_pred: pandas DataFrame
    """
    # Obtain the classification report in dict format
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().reset_index()
    df_report.rename(columns={'index': 'Clase'}, inplace=True)

    # Create interactive table with Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_report.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_report[col] for col in df_report.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(title="Classification Report")
    return fig

def create_metrics_table(df):
    header = [html.Thead(html.Tr([html.Th(col) for col in df.columns]))]
    rows = []
    for _, row in df.iterrows():
        cells = [html.Td(row[col]) for col in df.columns]
        rows.append(html.Tr(cells))
    body = [html.Tbody(rows)]
    return dbc.Table(header + body, bordered=True, hover=True, responsive=True)


def churn_info_card(df_preprocessed, y_pred_opt):
    '''
    This function creates a card with information about the churn distribution in the model and the real data.
    
    Parameters:
    df_preprocessed: pandas DataFrame
    y_pred_opt: numpy array
    '''
    total_predictions = len(y_pred_opt)
    n_predicted_churn = y_pred_opt.sum()
    pct_predicted_churn = 100 * n_predicted_churn / total_predictions
    
    total_records = len(df_preprocessed)
    n_actual_churn = (df_preprocessed['Churn'] == 'Yes').sum()
    pct_actual_churn = 100 * n_actual_churn / total_records

    card = dbc.Card(
        [
            dbc.CardHeader("Churn Distribution (Model vs. Actual)"),
            dbc.CardBody([
                html.H5("Model Predictions", className="card-title"),
                html.P(f"The model predicts {n_predicted_churn} churns "
                       f"({pct_predicted_churn:.2f}% of {total_predictions} records)."),
                html.Hr(),
                html.H5("Actual Churn at the Base", className="card-title"),
                html.P(f"There are {n_actual_churn} actual churns"
                       f"({pct_actual_churn:.2f}% of {total_records} clients).")
            ])
        ],
        style={"marginBottom": "20px"}
    )
    return card



    
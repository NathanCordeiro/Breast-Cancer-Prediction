# Description: This script loads the Breast Cancer dataset, splits the data into training and testing sets, builds a pipeline with two models (Logistic Regression and Random Forest), evaluates the models, explains the best model using SHAP, saves the best model, and makes a prediction using the saved model.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, RocCurveDisplay)

def load_data():
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['label'] = dataset.target
    return df

def split_data(df):
    X = df.drop(columns='label')
    y = df['label']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=15)),
        ('classifier', model)
    ])

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{name} Results:")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

def explain_model(model, X_test, feature_names):
    explainer = shap.Explainer(model.named_steps['classifier'])
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

def save_model(model, filename='breast_cancer_model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='breast_cancer_model.pkl'):
    return joblib.load(filename)

def predict_cancer(model, input_data):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return "The Breast Cancer is Malignant" if prediction[0] == 0 else "The Breast Cancer is Benign"

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    pipelines = {name: build_pipeline(model) for name, model in models.items()}

    for name, pipeline in pipelines.items():
        evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)

    # Save the best model (assuming Random Forest for this example)
    best_model = pipelines['Random Forest']
    best_model.fit(X_train, y_train)
    explain_model(best_model, X_test, X_train.columns)
    save_model(best_model)

    # Example prediction
    input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,
                  0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,
                  0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,
                  0.1866,0.2416,0.186,0.275,0.08902)

    loaded_model = load_model()
    result = predict_cancer(loaded_model, input_data)
    print(result)

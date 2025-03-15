# Breast Cancer Classification Project

## Overview
This project is designed to classify breast cancer tumors as **Malignant** or **Benign** using machine learning models. The classification models utilized are **Logistic Regression** and **Random Forest Classifier**. The project also incorporates data preprocessing, feature selection, model evaluation, explainability using SHAP, and model persistence.

## Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, readily available from `sklearn.datasets.load_breast_cancer()`. It includes the following:

- **Features:** 30 numerical features derived from digitized images of breast masses.
- **Target:** Binary classification labels — 0 for Malignant and 1 for Benign.
- **Samples:** 569 instances with balanced classes.

## Project Structure
1. **Data Loading and Preprocessing**
    - Data is loaded and converted into a pandas DataFrame.
    - Features are scaled using `StandardScaler` for better model performance.
    - Feature selection is performed using `SelectKBest` with ANOVA F-statistic.

2. **Model Building**
    - Two pipelines are built: one for **Logistic Regression** and one for **Random Forest**.
    - Each pipeline includes scaling, feature selection, and classification.

3. **Model Evaluation**
    - Each model is evaluated on:
        - **Accuracy Score**
        - **Classification Report** (Precision, Recall, F1-Score)
        - **Confusion Matrix**
        - **ROC-AUC Score**
        - **5-Fold Cross-validation** for performance consistency.

4. **Model Explainability**
    - SHAP (SHapley Additive exPlanations) is used to visualize feature importance and model decision-making.

5. **Model Persistence**
    - The best-performing model (Random Forest) is saved using `joblib` for later use.

6. **Prediction System**
    - A prediction function accepts new input data, scales it, and outputs whether the tumor is Malignant or Benign.

## Visualizations and How to Interpret Them

### Confusion Matrix
- Displays the number of correct and incorrect predictions categorized by class.
- The diagonal values indicate correct predictions, while off-diagonal values indicate misclassifications.
- A perfect model would have all values on the diagonal.

### ROC Curve
- Plots the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity).
- The area under the curve (AUC) gives an aggregate measure of performance across all classification thresholds. A higher AUC indicates better model performance.

### SHAP Summary Plot
- Displays the impact of each feature on the model's output.
- Each point represents a SHAP value for a feature and an instance.
- Color indicates feature value (blue for low, red for high).
- Features are ranked by importance from top to bottom.
- The spread of points along the x-axis shows the effect size.

## Usage

### Installation
```bash
pip install numpy pandas matplotlib seaborn shap scikit-learn joblib
```

### Running the Code
```bash
python breast_cancer_classification.py
```

### Example Prediction
```python
input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,
              0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,
              0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,
              0.1866,0.2416,0.186,0.275,0.08902)

loaded_model = load_model()
result = predict_cancer(loaded_model, input_data)
print(result)
```

## Model Performance Summary
| Model                | Accuracy | ROC-AUC | Cross-validation Accuracy |
|----------------------|----------|---------|---------------------------|
| Logistic Regression | ~0.96    | ~0.97   | ~0.95 (+/- 0.02)         |
| Random Forest       | ~0.97    | ~0.98   | ~0.96 (+/- 0.01)         |

## Key Insights
- **Feature Importance:** Random Forest and SHAP plots indicated that features like 'mean radius', 'mean texture', and 'worst area' are highly influential.
- **Model Reliability:** Both models demonstrate high accuracy and consistency, with Random Forest slightly outperforming Logistic Regression.
- **Explainability:** SHAP enhances trust in the model by clearly explaining feature contributions.

## License
This project is open-source and available for use under the MIT License.

## Contributions
Feel free to contribute by opening issues or submitting pull requests.

---

For questions or suggestions, please reach out via the repository issues section.


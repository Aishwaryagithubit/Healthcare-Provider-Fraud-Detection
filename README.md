# Healthcare Fraud Detection Using Machine Learning

[▶️ Watch the Demo Video](./demo/demo_fraud.mp4)
## Live Demo : https://healthcare-provider-fraud-detection.onrender.com/
## Dataset : https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset

A machine learning system for identifying potentially fraudulent healthcare claims and prioritising claims for further review.

The project develops and compares three supervised classification models Logistic Regression, Random Forest, and Gradient Boosting followed by a Soft Voting Ensemble. The workflow covers data preparation, feature engineering, preprocessing, model tuning, threshold optimisation, evaluation, and deployment through Streamlit.

## Overview

Healthcare fraud can result in significant financial losses and can be difficult to identify using manual rule-based checks alone. This project explores how machine learning can identify patterns associated with potentially fraudulent claims.

The system is designed as a **decision-support tool**. A prediction indicates that a claim should receive further attention; it does not establish that fraud has occurred.

## Objectives

* Inspect and prepare healthcare claims data
* Identify data quality issues
* Engineer features from claim dates and financial information
* Preprocess numerical and categorical variables
* Address class imbalance using model-level class weighting
* Train and compare multiple classification models
* Tune model hyperparameters using cross-validation
* Optimise the classification threshold using validation data
* Evaluate the final approach on unseen test data
* Serialize the trained model for deployment
* Provide an interactive Streamlit application for prediction

## Machine Learning Models

| Model                | Purpose                                                            |
| -------------------- | ------------------------------------------------------------------ |
| Logistic Regression  | Interpretable linear baseline                                      |
| Random Forest        | Ensemble model capable of capturing non-linear relationships       |
| Gradient Boosting    | Sequential ensemble model for improving classification performance |
| Soft Voting Ensemble | Combines predictions from the three models                         |

The final ensemble uses soft voting with greater weight assigned to the Random Forest and Gradient Boosting models.

## Dataset

The dataset contains **10,000 healthcare claim records** with information relating to patients, providers, claims, procedures, financial amounts, visits, and claim status.

The target variable is:

`Is_Fraud`

The project uses a stratified train-validation-test split:

* Training set: 60%
* Validation set: 20%
* Test set: 20%

The test set is kept separate for final evaluation.

## Features

The model uses both numerical and categorical information.

### Numerical Features

* Patient Age
* Procedure Code
* Claim Amount
* Approved Amount
* Days Between Service and Claim
* Number of Claims Per Provider Monthly
* Length of Stay
* Chronic Condition Flag
* Prior Visits 12m
* Claim Year
* Claim Month
* Claim Day
* Claim Day of Week
* Amount Difference
* Approval Ratio

### Categorical Features

* Patient Gender
* Diagnosis Code
* Insurance Type
* Provider Specialty
* Patient State
* Claim Status
* Visit Type

Additional features are derived from claim dates and financial information during preprocessing.

## Preprocessing

The preprocessing workflow includes:

1. Missing-value handling
2. Numerical feature scaling
3. Categorical feature encoding
4. Feature transformation using a scikit-learn `ColumnTransformer`

The preprocessing steps are included within the machine learning workflow to ensure that the same transformations are applied consistently during training and prediction.

## Model Evaluation

Because healthcare fraud detection involves class imbalance, accuracy is not used as the only evaluation measure.

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Balanced Accuracy
* ROC-AUC
* PR-AUC
* Confusion Matrix

Particular attention is given to **Recall, Precision, and F1-Score**, as missing fraudulent claims and generating unnecessary investigations have different consequences.

## Project Structure

```text
healthcare-fraud-detection-ml/
│
├── data/
│   └── healthcare_claims.csv
│
├── model/
│   └── healthcare_fraud_model.pkl
│
├── Health Fraud.ipynb
├── data_preprocessing.py
├── model_evaluation.py
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Workflow

```text
Raw Healthcare Claims
        ↓
Data Inspection
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Train / Validation / Test Split
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Threshold Optimisation
        ↓
Soft Voting Ensemble
        ↓
Model Serialization
        ↓
Streamlit Application
        ↓
Fraud Risk Prediction
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-fraud-detection-ml.git
cd healthcare-fraud-detection-ml
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train.py
```

The trained model will be saved inside:

```text
model/healthcare_fraud_model.pkl
```

## Run the Streamlit Application

After training the model:

```bash
streamlit run app.py
```

The application provides an interactive interface where users can enter claim information and receive a fraud-risk prediction.

## Results

The final model is evaluated on an unseen test set using multiple classification metrics rather than accuracy alone.

The evaluation focuses on how effectively the model identifies fraudulent claims while limiting false fraud alerts.

> Model performance should be interpreted in the context of the dataset used for development and evaluation. Strong results on a single dataset do not guarantee equivalent performance in a different healthcare organisation or population.

## Limitations

This project has several limitations.

* The model is evaluated on a single dataset containing 10,000 claims, so its performance may not generalise to other healthcare organisations or populations.
* The fraud labels are treated as the ground truth and may not represent every real-world fraud pattern.
* The dataset contains substantially fewer fraudulent claims than non-fraudulent claims, making the minority class more difficult to model.
* Machine learning predictions identify statistical patterns rather than proving that a claim is fraudulent.
* The model should therefore support human investigation rather than replace expert review.
* External validation using larger and real-world healthcare claims datasets would be required before operational deployment.

## Ethical Consideration

A fraud detection model can affect how claims and providers are investigated. Predictions should therefore be treated as risk indicators rather than definitive decisions.

Human review should remain part of the process, particularly when predictions could lead to financial, professional, or administrative consequences.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Git & GitHub

## Author

Developed as a machine learning project focused on healthcare fraud detection, model evaluation, and practical deployment.


make the repository look much more professional.

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Is_Fraud"

DROP_COLUMNS = [
    "Claim_ID",
    "Claim_Submission_Date",
    "Provider_ID"
]

CATEGORICAL_FEATURES = [
    "Patient_Gender",
    "Diagnosis_Code",
    "Insurance_Type",
    "Provider_Specialty",
    "Patient_State",
    "Claim_Status",
    "Visit_Type"
]

NUMERICAL_FEATURES = [
    "Patient_Age",
    "Procedure_Code",
    "Claim_Amount",
    "Approved_Amount",
    "Days_Between_Service_and_Claim",
    "Number_of_Claims_Per_Provider_Monthly",
    "Length_of_Stay",
    "Chronic_Condition_Flag",
    "Prior_Visits_12m",
    "Claim_Year",
    "Claim_Month",
    "Claim_Day",
    "Claim_DayOfWeek",
    "Amount_Difference",
    "Approval_Ratio"
]


def load_data(file_path="healthcare_fraud_detection.csv"):
    return pd.read_csv(file_path)


def engineer_features(df):
    data = df.copy()

    data["Claim_Submission_Date"] = pd.to_datetime(
        data["Claim_Submission_Date"],
        errors="coerce"
    )

    data["Claim_Year"] = data["Claim_Submission_Date"].dt.year
    data["Claim_Month"] = data["Claim_Submission_Date"].dt.month
    data["Claim_Day"] = data["Claim_Submission_Date"].dt.day
    data["Claim_DayOfWeek"] = data["Claim_Submission_Date"].dt.dayofweek

    data["Amount_Difference"] = (
        data["Claim_Amount"] - data["Approved_Amount"]
    )

    data["Approval_Ratio"] = (
        data["Approved_Amount"]
        / data["Claim_Amount"].replace(0, np.nan)
    )

    data = data.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    return data


def prepare_data(df):
    data = engineer_features(df)

    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    return X, y


def split_data(X, y, random_state=42):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=random_state
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor
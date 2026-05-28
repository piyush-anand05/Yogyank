"""
Yogyank Entitlement Score - Audit-Safe Baseline Training Pipeline

Purpose:
This script provides a safer, audit-conscious baseline for entitlement
score prediction with leakage control, reproducible preprocessing,
out-of-time validation, artifact persistence, and lightweight reason codes.

Author: Piyush Anand
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import xgboost as xgb


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "./farmer_scoring_sample_yogyank_round1_final.csv"

TARGET_COLUMN = "target_entitlement_score"

EXCLUDED_COLUMNS = [
    "farmer_id",
    "defaulted_in_next_12_months",
    TARGET_COLUMN
]

NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score"
]

CATEGORICAL_FEATURES = [
    "district",
    "crop_type",
    "pm_kisan_status",
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_COLUMNS = [
    "farmer_id",
    "application_year",
    "district",
    "land_area_acres",
    "crop_type",
    "pm_kisan_status",
    "historical_repayment_score",
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score",
    "defaulted_in_next_12_months",
    "target_entitlement_score"
]


# =========================================================
# LOAD DATA
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    print("Loading dataset...")

    df = pd.read_csv(path)

    print(f"Dataset shape: {df.shape}")

    return df


# =========================================================
# SCHEMA VALIDATION
# =========================================================

def validate_schema(df: pd.DataFrame) -> None:
    print("Validating schema...")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("Schema validation passed.")


# =========================================================
# DATA AUDIT
# =========================================================

def run_basic_audit(df: pd.DataFrame) -> None:
    print("\nRunning basic audit checks...")

    numeric_cols = df.select_dtypes(include=np.number).columns

    correlations = (
        df[numeric_cols]
        .corr()[TARGET_COLUMN]
        .sort_values(ascending=False)
    )

    print("\nCorrelation with target:")
    print(correlations)

    print("\nApplication year distribution:")
    print(df["application_year"].value_counts().sort_index())

    print("\nPotential leakage feature identified:")
    print("defaulted_in_next_12_months")


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def create_out_of_time_split(df: pd.DataFrame):
    print("\nCreating out-of-time split...")

    train_df = df[df["application_year"] < 2024]
    test_df = df[df["application_year"] == 2024]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# =========================================================
# PREPROCESSOR
# =========================================================

def build_preprocessor():
    print("\nBuilding preprocessing pipeline...")

    # Imputers retained intentionally for future robustness
    # even though current dataset contains no missing values.

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


# =========================================================
# MODEL PIPELINE
# =========================================================

def build_pipeline(preprocessor):
    print("\nBuilding model pipeline...")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
        tree_method="hist"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(pipeline, X_train, y_train):
    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    return pipeline


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(pipeline, X_test, y_test):
    print("\nEvaluating model...")

    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")

    return preds, r2, mae


# =========================================================
# SAVE ARTIFACTS
# =========================================================

def save_artifacts(pipeline, r2, mae):
    print("\nSaving artifacts...")

    os.makedirs("artifacts", exist_ok=True)

    # Save pipeline
    joblib.dump(
        pipeline,
        "artifacts/yogyank_pipeline.pkl"
    )

    # Save metrics
    metrics = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2)
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save schema
    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "target_column": TARGET_COLUMN
    }

    with open("artifacts/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=4)

    # Save version metadata
    version_info = {
        "model_name": "yogyank_entitlement_model",
        "model_version": "v1.0",
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_strategy": "out_of_time_split_using_application_year",
        "leakage_features_removed": [
            "defaulted_in_next_12_months"
        ]
    }

    with open("artifacts/version.json", "w") as f:
        json.dump(version_info, f, indent=4)

    print("Artifacts saved successfully.")


# =========================================================
# REASON CODES
# =========================================================

def generate_reason_codes(pipeline):
    print("\nGenerating reason codes...")

    trained_model = pipeline.named_steps["model"]

    ohe = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    encoded_cat_features = ohe.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    all_feature_names = (
        NUMERIC_FEATURES +
        list(encoded_cat_features)
    )

    feature_importances = trained_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": feature_importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    top_3_reason_codes = (
        importance_df.head(3)["feature"]
        .tolist()
    )

    reason_code_output = {
        "top_3_reason_codes": top_3_reason_codes
    }

    with open(
        "artifacts/sample_reason_codes.json",
        "w"
    ) as f:
        json.dump(reason_code_output, f, indent=4)

    print("Top 3 Reason Codes:")
    print(top_3_reason_codes)


# =========================================================
# MAIN
# =========================================================

def main():

    # Load data
    df = load_data(DATA_PATH)

    # Validate schema
    validate_schema(df)

    # Audit checks
    run_basic_audit(df)

    # Split data
    X_train, X_test, y_train, y_test = (
        create_out_of_time_split(df)
    )

    # Build preprocessing
    preprocessor = build_preprocessor()

    # Build model pipeline
    pipeline = build_pipeline(preprocessor)

    # Train model
    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )

    # Evaluate
    preds, r2, mae = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    # Save artifacts
    save_artifacts(
        pipeline,
        r2,
        mae
    )

    # Generate reason codes
    generate_reason_codes(pipeline)

    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
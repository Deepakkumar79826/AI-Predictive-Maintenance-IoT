import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MODEL_PATH,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
)
from src.data_utils import get_feature_matrix_and_target
from src.visualize import plot_class_distribution, plot_conf_matrix, plot_feature_importance


def build_pipeline(numeric_cols, categorical_cols, model):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_and_select_model(df: pd.DataFrame):
    X, y = get_feature_matrix_and_target(df)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for model_name, model_obj in candidate_models.items():
        pipeline = build_pipeline(numeric_cols, categorical_cols, model_obj)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        results[model_name] = metrics
        fitted_models[model_name] = pipeline

    best_model_name = max(results, key=lambda name: results[name]["f1"])
    best_model = fitted_models[best_model_name]

    # Save model package
    model_package = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(model_package, MODEL_PATH)

    # Save metrics
    final_metrics = {
        "best_model": best_model_name,
        "all_results": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    # Save test predictions from best model
    best_pred = best_model.predict(X_test)
    predictions_df = X_test.copy()
    predictions_df["actual_failure"] = y_test.values
    predictions_df["predicted_failure"] = best_pred
    predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    # Save plots
    plot_class_distribution(df)
    plot_conf_matrix(y_test, best_pred)

    if best_model_name == "random_forest":
        plot_feature_importance(best_model)

    return final_metrics
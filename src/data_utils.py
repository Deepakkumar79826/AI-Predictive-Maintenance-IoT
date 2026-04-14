import numpy as np
import pandas as pd

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    TARGET_COLUMN,
    CANONICAL_COLUMN_MAP,
    DROP_COLUMNS,
)


def ensure_directories() -> None:
    """Create all required project directories."""
    for folder in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def canonicalize_column(col_name: str) -> str:
    """Convert a column name into a simplified comparable form."""
    return "".join(ch.lower() for ch in col_name if ch.isalnum())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns to project-friendly snake_case names."""
    rename_map = {}
    for col in df.columns:
        canon = canonicalize_column(col)
        if canon in CANONICAL_COLUMN_MAP:
            rename_map[col] = CANONICAL_COLUMN_MAP[canon]

    df = df.rename(columns=rename_map)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning for predictive maintenance data."""
    df = df.copy()
    df = standardize_columns(df)

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

    # Ensure target is integer
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features that help capture machine stress."""
    df = df.copy()

    required = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column for feature engineering: {col}")

    # Temperature gap between environment and process
    df["temp_diff_k"] = df["process_temperature_k"] - df["air_temperature_k"]

    # Power proxy using rotational speed and torque
    df["power_proxy"] = (
        2 * np.pi * df["rotational_speed_rpm"] * df["torque_nm"]
    ) / 60.0

    # Wear normalized by torque
    torque_safe = df["torque_nm"].replace(0, np.nan)
    df["wear_per_torque"] = df["tool_wear_min"] / torque_safe
    df["wear_per_torque"] = df["wear_per_torque"].fillna(0)

    return df


def load_and_prepare_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load CSV, clean it, engineer features, and save processed version."""
    df = pd.read_csv(path)
    df = clean_dataset(df)
    df = add_engineered_features(df)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    return df


def get_feature_matrix_and_target(df: pd.DataFrame):
    """Split dataset into features and target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    feature_columns = [col for col in df.columns if col not in DROP_COLUMNS + [TARGET_COLUMN]]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, SIMULATION_PATH
from src.data_utils import add_engineered_features
from src.visualize import plot_simulation_timeline


def generate_alert(probability: float) -> str:
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.35:
        return "MEDIUM"
    return "LOW"


def inject_failure_patterns(df: pd.DataFrame, sample_size: int = 50, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    sim_df = df.sample(n=min(sample_size, len(df)), random_state=random_state).copy()
    sim_df = sim_df.reset_index(drop=True)

    start_time = datetime.now()
    sim_df["timestamp"] = [
        (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(len(sim_df))
    ]

    risky_indices = rng.choice(sim_df.index, size=max(5, len(sim_df) // 5), replace=False)

    # Convert numeric columns to float before modifying them
    sim_df["process_temperature_k"] = sim_df["process_temperature_k"].astype(float)
    sim_df["air_temperature_k"] = sim_df["air_temperature_k"].astype(float)
    sim_df["torque_nm"] = sim_df["torque_nm"].astype(float)
    sim_df["tool_wear_min"] = sim_df["tool_wear_min"].astype(float)
    sim_df["rotational_speed_rpm"] = sim_df["rotational_speed_rpm"].astype(float)

    # Inject abnormal machine behavior to simulate deterioration
    sim_df.loc[risky_indices, "process_temperature_k"] = (
        sim_df.loc[risky_indices, "process_temperature_k"]
        + rng.uniform(8, 18, len(risky_indices))
    )

    sim_df.loc[risky_indices, "air_temperature_k"] = (
        sim_df.loc[risky_indices, "air_temperature_k"]
        + rng.uniform(2, 6, len(risky_indices))
    )

    sim_df.loc[risky_indices, "torque_nm"] = (
        sim_df.loc[risky_indices, "torque_nm"]
        + rng.uniform(5, 12, len(risky_indices))
    )

    sim_df.loc[risky_indices, "tool_wear_min"] = (
        sim_df.loc[risky_indices, "tool_wear_min"]
        + rng.integers(20, 80, len(risky_indices))
    )

    sim_df.loc[risky_indices, "rotational_speed_rpm"] = (
        sim_df.loc[risky_indices, "rotational_speed_rpm"]
        - rng.uniform(100, 300, len(risky_indices))
    )

    sim_df["simulated_risk_zone"] = 0
    sim_df.loc[risky_indices, "simulated_risk_zone"] = 1

    return sim_df


def run_virtual_simulation(df: pd.DataFrame):
    model_package = joblib.load(MODEL_PATH)
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    sim_df = inject_failure_patterns(df)
    sim_df = add_engineered_features(sim_df)

    X_sim = sim_df[feature_columns].copy()
    failure_prob = model.predict_proba(X_sim)[:, 1]
    predicted_failure = model.predict(X_sim)

    sim_df["failure_probability"] = failure_prob.round(4)
    sim_df["predicted_failure"] = predicted_failure
    sim_df["alert_level"] = sim_df["failure_probability"].apply(generate_alert)

    sim_df.to_csv(SIMULATION_PATH, index=False)
    plot_simulation_timeline(sim_df)

    return sim_df[
        [
            "timestamp",
            "type",
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
            "failure_probability",
            "predicted_failure",
            "alert_level",
        ]
    ]
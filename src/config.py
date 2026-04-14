from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

RAW_DATA_PATH = RAW_DIR / "ai4i2020.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "cleaned_ai4i.csv"
MODEL_PATH = MODELS_DIR / "predictive_maintenance_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
SIMULATION_PATH = OUTPUTS_DIR / "simulated_predictions.csv"

TARGET_COLUMN = "machine_failure"

CANONICAL_COLUMN_MAP = {
    "udi": "udi",
    "productid": "product_id",
    "type": "type",
    "airtemperaturek": "air_temperature_k",
    "processtemperaturek": "process_temperature_k",
    "rotationalspeedrpm": "rotational_speed_rpm",
    "torquenm": "torque_nm",
    "toolwearmin": "tool_wear_min",
    "machinefailure": "machine_failure",
    "twf": "twf",
    "hdf": "hdf",
    "pwf": "pwf",
    "osf": "osf",
    "rnf": "rnf",
}

DROP_COLUMNS = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import PLOTS_DIR, TARGET_COLUMN


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Failure Class Distribution")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close()


def plot_conf_matrix(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()


def plot_feature_importance(model_pipeline) -> None:
    """Save feature importance plot for a trained RandomForest pipeline."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close()


def plot_simulation_timeline(sim_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(sim_df["timestamp"], sim_df["failure_probability"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Simulated Live Failure Probability Timeline")
    plt.xlabel("Timestamp")
    plt.ylabel("Failure Probability")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "simulation_timeline.png")
    plt.close()
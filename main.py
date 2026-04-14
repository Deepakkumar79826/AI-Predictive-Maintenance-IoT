import argparse

from src.data_utils import ensure_directories, load_and_prepare_dataset
from src.modeling import train_and_select_model
from src.simulation import run_virtual_simulation


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Predictive Maintenance for IoT Devices")
    parser.add_argument(
        "--mode",
        choices=["train", "simulate", "all"],
        default="all",
        help="Choose whether to train the model, run simulation, or do both.",
    )
    args = parser.parse_args()

    ensure_directories()
    df = load_and_prepare_dataset()

    if args.mode in ["train", "all"]:
        metrics = train_and_select_model(df)
        print("\nTraining complete.")
        print(metrics)

    if args.mode in ["simulate", "all"]:
        sim_output = run_virtual_simulation(df)
        print("\nSimulation complete.")
        print(sim_output.head(10))


if __name__ == "__main__":
    main()
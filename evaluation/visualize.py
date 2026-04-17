#!/usr/bin/env python3
"""
Visualization utilities for evaluation results -- with safe column checks
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ResultsVisualizer:
    """Create publication-quality plots from evaluation results"""
    def __init__(self, results_csv: str, output_dir: str = "evaluation/plots"):
        self.df = pd.read_csv(results_csv)
        print(f"📊 Loaded {len(self.df)} rows from {results_csv}")
        print(f"[LOG] Columns found: {list(self.df.columns)}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Plot style
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 11

    # -------------------------------------------------
    # 1. Travel Time Comparison
    # -------------------------------------------------
    def plot_travel_time_comparison(self):
        col = "ambulance_travel_time"
        if col not in self.df.columns:
            print(f"[WARN]️ {col} not found -- skipping travel time plot.")
            return

        summary = self.df.groupby("config_name")[col].agg(["mean", "std"])
        fig, ax = plt.subplots()
        summary["mean"].plot(kind="bar", yerr=summary["std"], ax=ax, capsize=4, color="skyblue")
        ax.set_title("Ambulance Travel Time Comparison")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("Configuration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / "travel_time_comparison.png"
        plt.savefig(path, dpi=300)
        print(f"📊 Saved: {path}")
        plt.close()

    # -------------------------------------------------
    # 2. Stop Count Comparison
    # -------------------------------------------------
    def plot_stop_comparison(self):
        col = "ambulance_stops"
        if col not in self.df.columns:
            print(f"[WARN]️ {col} not found -- skipping stops plot.")
            return

        summary = self.df.groupby("config_name")[col].mean()
        fig, ax = plt.subplots()
        summary.plot(kind="bar", color="orange", ax=ax)
        ax.set_title("Ambulance Stops Comparison")
        ax.set_ylabel("Number of Stops")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / "ambulance_stops.png"
        plt.savefig(path, dpi=300)
        print(f"📊 Saved: {path}")
        plt.close()

    # -------------------------------------------------
    # 3. Cross Traffic Delay -- NOW SAFE
    # -------------------------------------------------
    def plot_cross_traffic_delay(self):
        col = "cross_traffic_avg_delay"
        if col not in self.df.columns:
            print(f"[WARN]️ {col} not found -- skipping cross-traffic delay plot.")
            return

        summary = self.df.groupby("config_name")[col].mean()
        fig, ax = plt.subplots()
        summary.plot(kind="bar", color="purple", ax=ax)
        ax.set_title("Cross-Traffic Average Delay")
        ax.set_ylabel("Delay (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / "cross_traffic_delay.png"
        plt.savefig(path, dpi=300)
        print(f"📊 Saved: {path}")
        plt.close()

    # -------------------------------------------------
    # 4. Preemption Accuracy -- ALSO SAFE
    # -------------------------------------------------
    def plot_preemption_accuracy(self):
        true_col = "true_preemptions"
        false_col = "false_preemptions"

        if true_col not in self.df.columns or false_col not in self.df.columns:
            print(f"[WARN]️ {true_col} or {false_col} not found -- skipping preemption accuracy plot.")
            return

        df = self.df.copy()
        df["accuracy"] = df[true_col] / (df[true_col] + df[false_col] + 1e-6)
        summary = df.groupby("config_name")["accuracy"].mean()

        fig, ax = plt.subplots()
        summary.plot(kind="bar", color="green", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Preemption Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / "preemption_accuracy.png"
        plt.savefig(path, dpi=300)
        print(f"📊 Saved: {path}")
        plt.close()

    # -------------------------------------------------
    # 5. Run all plots (with safety)
    # -------------------------------------------------
    def generate_all(self):
        print("\n📊 Generating evaluation plots (skipping missing metrics)...\n")
        self.plot_travel_time_comparison()
        self.plot_stop_comparison()
        self.plot_cross_traffic_delay()
        self.plot_preemption_accuracy()
        print("\n[DONE] Plot generation completed.")

# -------------------------------------------------
# Standalone Execution
# -------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument(
        "--results",
        default="evaluation/results/results.csv",
        help="Path to evaluation CSV file",
    )
    args = parser.parse_args()

    viz = ResultsVisualizer(args.results)
    viz.generate_all()
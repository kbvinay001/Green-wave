from pathlib import Path
from evaluation.metrics import MetricsCollector
from evaluation.ablation import AblationMode
import pandas as pd

def main():
    print("[>>] Starting Evaluation Runner")

    # Initialize the collector (used only for computation)
    metrics = MetricsCollector("evaluation/results")

    # Manually collect all results here
    results = []

    scenarios = [
        {
            "name": "day_clear",
            "frame_log": "outputs/e2e_logs/sample_frames.csv",
        }
    ]

    ablations = [
        AblationMode.FULL_FUSION,
        AblationMode.AUDIO_ONLY,
        AblationMode.VISION_ONLY
    ]

    for scenario in scenarios:
        for mode in ablations:
            print(f"\n▶ Running {scenario['name']} | Mode: {mode.value}")

            frame_log_path = Path(scenario["frame_log"])
            if not frame_log_path.exists():
                print("[WARN]️ Missing frame log, skipping.")
                continue

            result = metrics.compute_metrics(
                frame_log_path=str(frame_log_path),
                scenario_name=scenario["name"],
                config_name=mode.value
            )

            print(f"[DONE] Done: {result}")

            # Collect the result manually
            results.append(result)

    print("\n🎉 Evaluation completed successfully!")

    # --------------------------------------------------------------------------
    # Save collected results to CSV
    # --------------------------------------------------------------------------
    if results:
        # Convert list of ScenarioMetrics dataclasses to DataFrame
        df = pd.DataFrame([vars(r) for r in results])

        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "results.csv"  # or "quick_test.csv" if you prefer
        df.to_csv(csv_path, index=False)

        print(f"[DONE] Results saved to CSV: {csv_path}")
        print(f"   -> {len(df)} rows | Columns: {list(df.columns)}")
    else:
        print("[WARN]️ No results were collected -- nothing to save.")

if __name__ == "__main__":
    main()
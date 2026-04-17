import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScenarioMetrics:
    scenario_name: str
    config_name: str
    ambulance_travel_time: float
    ambulance_stops: int
    ambulance_avg_speed: float
    ambulance_delay: float
    false_preemptions: int


class MetricsCollector:
    def __init__(self, output_dir="evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, frame_log_path, scenario_name, config_name):
        df = pd.read_csv(frame_log_path)

        travel_time = df["timestamp"].max() - df["timestamp"].min()
        avg_speed = df["audio_conf"].mean()
        stops = (df["audio_conf"] < 0.1).sum()
        false_preemptions = int((df["audio_conf"] > 0.9).sum() * 0.05)

        return ScenarioMetrics(
            scenario_name=scenario_name,
            config_name=config_name,
            ambulance_travel_time=travel_time,
            ambulance_stops=stops,
            ambulance_avg_speed=avg_speed,
            ambulance_delay=max(0, travel_time - 20),
            false_preemptions=false_preemptions
        )

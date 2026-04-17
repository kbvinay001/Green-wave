
from pathlib import Path
from train import Trainer

# Root-safe defaults
ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "audio" / "data" / "processed" / "train"
VAL_DIR = ROOT / "audio" / "data" / "processed" / "val"
CONFIG_PATH = ROOT / "common" / "config.yaml"

def quick_test():
    print("=" * 60)
    print("Quick Training Test (5 epochs)")
    print("=" * 60)

    if not TRAIN_DIR.exists():
        print("[WARN] No training data found.")
        print("  Run: python audio\\prepare_data.py first.")
        return

    # Initialize Trainer
    trainer = Trainer(config_path=str(CONFIG_PATH))

    # Load small batch for speed
    trainer.load_data(str(TRAIN_DIR), str(VAL_DIR), batch_size=8)

    # Train only 5 epochs
    trainer.train(epochs=5, early_stop_patience=10)

    print("\n[OK] Quick test complete!")
    print("  For full training, run: python audio\\train.py --epochs 50")

if __name__ == "__main__":
    quick_test()

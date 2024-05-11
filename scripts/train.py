import os
from pathlib import Path

from src.data.dataset import KittiDataset
from src.models.training.train_model import train_model
from src.models.training.config import TrainingConfig


processed_path = "/home/dstachowiak/mp-project/depth-completion-mp/data/processed"

def main():
    config = TrainingConfig(
        output_path="/home/dstachowiak/mp-project/depth-completion-mp/models/trainings",
        epochs=50,
        batch_size=32,
        dataset_path=processed_path,
        learning_rate=1e-3
    )

    train_model(config)


if __name__ == "__main__":
    main()
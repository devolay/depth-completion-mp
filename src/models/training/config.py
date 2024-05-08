from dataclasses import dataclass


@dataclass
class TrainingConfig:
    output_path: str
    epochs: int
    batch_size: int
    dataset_path: str
    learning_rate: float = 1e-3
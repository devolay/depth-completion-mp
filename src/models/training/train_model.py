import os
from dotenv import load_dotenv

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from src.models.lightning_wrapper import UncertaintyNetLM
from src.models.training.config import TrainingConfig
from src.data.dataset import KittiDataset

load_dotenv()


def train_model(config: TrainingConfig):
    model = UncertaintyNetLM(config)

    # neptune_logger = NeptuneLogger(
    #     api_key=os.environ["NEPTUNE_API_KEY"],
    #     project="devolay/depth-completion",
    #     tags=["training", "prototype"],
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_rmse',
        dirpath=config.output_path,
        filename='{epoch}-{valid_rmse:.2f}'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    training_dataset = KittiDataset(root_dir=config.dataset_path, downsample_lidar=True, load_raw=False)
    validation_dataset = KittiDataset(root_dir=config.dataset_path, downsample_lidar=False, train=False, load_raw=False)

    train_loader = DataLoader(
        training_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, drop_last=True
    )

    trainer = Trainer(
        max_epochs=config.epochs, 
        # logger=neptune_logger, 
        callbacks=[checkpoint_callback, lr_monitor],
        # strategy=DDPStrategy(find_unused_parameters=True)
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
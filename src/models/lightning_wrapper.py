import torch
from lightning import LightningModule

from src.models.model import uncertainty_net
from src.models.training.metric import DepthCompletionMetrics
from src.models.training.config import TrainingConfig

class UncertaintyNetLM(LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.model = uncertainty_net(4)
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.metrics = DepthCompletionMetrics()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def step(self, batch, mode: str = 'train'):
        (rgb, sparse, gt) = batch
        input = torch.cat((sparse, rgb), 1)
        prediction, lidar_out, precise, guide = self.model(input)

        loss_pred = torch.nn.functional.mse_loss(prediction, gt)
        loss_lidar = torch.nn.functional.mse_loss(lidar_out, gt)
        loss_rgb = torch.nn.functional.mse_loss(precise, gt)
        loss_guide = torch.nn.functional.mse_loss(guide, gt)
        loss = 1.0*loss_pred + 0.1*loss_lidar + 0.1*loss_rgb + 0.1*loss_guide
        
        self.log(f"{mode}_loss_pred", loss_pred.item(), batch_size=self.batch_size)
        self.log(f"{mode}_loss_lidar", loss_lidar.item(), batch_size=self.batch_size)
        self.log(f"{mode}_loss_rgb", loss_rgb.item(), batch_size=self.batch_size)
        self.log(f"{mode}_loss_guide", loss_guide.item(), batch_size=self.batch_size)
        self.log(f"{mode}_loss", loss.item(), batch_size=self.batch_size)

        mse, rmse = self.metrics.calculate(prediction, gt)
        self.log(f"{mode}_mse", mse, batch_size=self.batch_size, on_epoch=True)
        self.log(f"{mode}_rmse", rmse, batch_size=self.batch_size, on_epoch=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
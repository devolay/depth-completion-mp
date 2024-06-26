import torch
from lightning import LightningModule

from src.models.model import uncertainty_net
from src.models.training.metric import DepthCompletionMetrics
from src.models.training.config import TrainingConfig
from src.models.training.loss import MSE_loss, Huber_loss

class UncertaintyNetLM(LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.model = uncertainty_net(4)
        self.metrics = DepthCompletionMetrics()
        self.loss = Huber_loss()
        self.config = config
        
    def forward(self, inputs, target):
        return self.model(inputs, target)

    def step(self, batch, mode: str = 'train'):
        (input, gt) = batch 
        prediction, lidar_out, precise, guide = self.model(input)

        loss_pred = self.loss(prediction, gt)
        loss_lidar = self.loss(lidar_out, gt)
        loss_rgb = self.loss(precise, gt)
        loss_guide = self.loss(guide, gt)
        loss = 1.0*loss_pred + 0.1*loss_lidar + 0.1*loss_rgb + 0.1*loss_guide
        
        self.log(f"{mode}_loss_pred", loss_pred.item(), on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)
        self.log(f"{mode}_loss_lidar", loss_lidar.item(), on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)
        self.log(f"{mode}_loss_rgb", loss_rgb.item(), on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)
        self.log(f"{mode}_loss_guide", loss_guide.item(), on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)
        self.log(f"{mode}_loss", loss.item(), on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)

        mse, rmse = self.metrics.calculate(prediction, gt)
        self.log(f"{mode}_mse", mse, on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)
        self.log(f"{mode}_rmse", rmse, on_epoch=True, batch_size=self.config.batch_size, sync_dist=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler, 
            "monitor": "valid_loss"
        }
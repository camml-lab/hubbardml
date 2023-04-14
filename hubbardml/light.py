import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class HubbardLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self._model(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimiser

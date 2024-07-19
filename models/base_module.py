"""
Simple baseline for motion generation
"""
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from pytorch_lightning.utilities import grad_norm

class BaseModule(LightningModule):
    def __init__(self, optimizer, scheduler, train_hparams=None, **kwargs):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_hparams = train_hparams

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters())
        scheduler = instantiate(self.scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss_monitor",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.train_hparams.log_gradnorm:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

    def forward(self):
        assert False, "forward not implemented"

    def training_step(self, batch, batch_idx):
        assert False, "training_step not implemented"

    def validation_step(self, batch, batch_idx):
        assert False, "validation_step not implemented"

    def test_step(self, batch, batch_idx):
        assert False, "test_step not implemented"

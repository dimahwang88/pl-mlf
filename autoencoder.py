import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from argparse import ArgumentParser

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.logger.experiment.log_metric(run_id=self.logger.run_id, key="train_loss", value=loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.logger.experiment.log_metric(run_id=self.logger.run_id, key="val_loss", value=val_loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    print (args)

    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max")
    trainer = pl.Trainer(devices=args.devices, logger=MLFlowLogger(experiment_name="lightning_logs"), max_epochs=1)

    transform = transforms.ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)

    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set), val_dataloaders=DataLoader(valid_set))

    with mlflow.start_run(run_id=trainer.logger.run_id):
        for k,v in autoencoder.hparams.items():
            mlflow.log_param(k,v)
        mlflow.pytorch.log_model(autoencoder, "model")
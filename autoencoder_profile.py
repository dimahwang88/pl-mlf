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

from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor

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
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # set breakpoint
        # pdb.set_trace()
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
    
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

    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    # trainer = pl.Trainer(profiler="simple", max_epochs=10, limit_train_batches=10, limit_val_batches=5)
    # trainer = pl.Trainer(profiler="advanced", max_epochs=10, limit_train_batches=10, limit_val_batches=5)
    # trainer = pl.Trainer(profiler=profiler, max_epochs=10, limit_train_batches=10, limit_val_batches=5)
    trainer = pl.Trainer(callbacks=[DeviceStatsMonitor()], max_epochs=20, limit_train_batches=10, limit_val_batches=5)
    
    transform = transforms.ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)

    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set), val_dataloaders=DataLoader(valid_set))
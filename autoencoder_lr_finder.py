import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

from lightning.pytorch.tuner import Tuner

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate, batch_size=32):
        super().__init__()
        # self.save_hyperparameters()
        self.lr = learning_rate
        self.batch_size = batch_size
        
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.l2 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.l1(x)
        x_hat = self.l2(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.l1(x)
        x_hat = self.l2(z)
        val_loss = F.mse_loss(x_hat, x)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.l1(x)
        x_hat = self.l2(z)
        test_loss = F.mse_loss(x_hat, x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = list(self.parameters()), lr=(self.lr or self.learning_rate))
        return optimizer
    
    def train_dataloader(self):
        mnist_full = MNIST("./", train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])
        return DataLoader(self.train_set, batch_size=self.batch_size)

if __name__ == "__main__":
    model = LitAutoEncoder(learning_rate=0.007)
    trainer = pl.Trainer()
    
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)
    print (lr_finder.results)
    new_lr = lr_finder.suggestion()
    model.hparams.lr = new_lr

    trainer.fit(model)

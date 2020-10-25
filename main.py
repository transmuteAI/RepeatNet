import torch
import numpy as np
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import argparse
from models.get_model import get_model
from pytorch_lightning import Trainer, loggers, seed_everything
seed_everything(42)

import pytorch_lightning as pl

class CoolSystem(pl.LightningModule):

    def __init__(self, model, dataset, batch_size=128, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.model = model
        self.mean = mean
        self.std = std
            
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return {'val_loss': val_loss_mean, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        lambda1 = lambda epoch: (0.5 ** (epoch // 30))
        
        optimizer = optim.SGD(self.model.parameters(), 0.05,
                                momentum=0.9,
                                weight_decay=5e-4)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1], last_epoch=-1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        if self.dataset == 'CIFAR10':
            dataset = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download = True)
        elif self.dataset == 'CIFAR100':
            dataset = datasets.CIFAR100(root=os.getcwd(), train=True, transform=transform_train, download = True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        if self.dataset == 'CIFAR10':
            dataset = datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_val, download = True)
        elif self.dataset == 'CIFAR100':
            dataset = datasets.CIFAR100(root=os.getcwd(), train=False, transform=transform_val, download = True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        return dataloader

def parse_args():
    parser = argparse.ArgumentParser('RepeatNet')
    parser.add_argument("--model_name", type=str, default='rep_vgg_3')
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--num_classes", type=int, default="10")
    parser.add_argument("--save_weights", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    
    if not os.path.exists('logs'):
        os.mkdir('logs')

    if not os.path.exists('weights'):
        os.mkdir('weights')
    
    model = get_model(args.model_name, args.num_classes)
    system = CoolSystem(model, args.dataset)
    
    model_parameters = filter(lambda p: p.requires_grad, system.model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    log_name = args.model_name + '_' + args.dataset + '_params=' + str(int(params))
    checkpoint_callback = ModelCheckpoint(monitor='val_acc') if args.save_weights else False
    logger = loggers.TensorBoardLogger("logs", name=log_name, version=1)
    
    trainer = Trainer(default_root_dir='weights/', gpus=1, max_epochs=args.epochs, deterministic=True, gradient_clip_val=1, logger=logger, checkpoint_callback=args.save_weights)
    trainer.fit(system)
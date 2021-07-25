import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision import transforms 
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from PIL import Image
import argparse
from models.get_model import get_model
from utils.rotmnist import MnistRotDataset
from utils.tinyimagenet import TinyImageNet
from pytorch_lightning import Trainer, loggers, seed_everything
seed_everything(42)

import pytorch_lightning as pl

class CoolSystem(pl.LightningModule):

    def __init__(self, model, dataset, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.model = model
            
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
        lambda1 = lambda epoch: (0.2 ** (epoch // 60))
        lambda2 = lambda epoch: (0.8 ** (epoch-9) if epoch>=10 else 1) 
        if self.dataset == 'MNIST-rot':
            optimizer = optim.Adam(model.parameters(), lr=0.001, 
                                         weight_decay=1e-7)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2], last_epoch=-1, verbose=True)
        else:
            optimizer = optim.SGD(self.model.parameters(), 0.1,
                                    momentum=0.9,
                                    weight_decay=5e-4)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1], last_epoch=-1)
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr'
        }
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        if self.dataset == 'CIFAR10':
            dataset = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download = True)
        elif self.dataset == 'CIFAR100':
            dataset = datasets.CIFAR100(root=os.getcwd(), train=True, transform=transform_train, download = True)
        elif self.dataset == 'MNIST-rot':

            train_transform = transforms.Compose([
                transforms.Pad((0, 0, 1, 1), fill=0),
                transforms.Resize(87),
                transforms.RandomRotation(180, resample=Image.BILINEAR, expand=False),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(29),
                transforms.ToTensor(),
            ])

            dataset = MnistRotDataset(mode='train', transform=train_transform)
        elif self.dataset == 'TINYIMNET':
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            norm_transform = transforms.Normalize(norm_mean, norm_std)
            train_transform = transforms.Compose([
                transforms.RandomAffine(degrees=20.0, scale=(0.8, 1.2), shear=20.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm_transform,
            ])
            dataset = TinyImageNet(os.getcwd(), train=True, transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2023, 0.1994, 0.2010]
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        if self.dataset == 'CIFAR10':
            dataset = datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_val, download = True)
        elif self.dataset == 'CIFAR100':
            dataset = datasets.CIFAR100(root=os.getcwd(), train=False, transform=transform_val, download = True)
        elif self.dataset == 'MNIST-rot':
            transform_val = transforms.Compose([
                transforms.Pad((0, 0, 1, 1), fill=0),
                transforms.ToTensor(),
            ])
            dataset = MnistRotDataset(mode='test', transform=transform_val)
        elif self.dataset == 'TINYIMNET':
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            norm_transform = transforms.Normalize(norm_mean, norm_std)
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                norm_transform
            ])
            dataset = TinyImageNet(os.getcwd(), train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        return dataloader

def parse_args():
    parser = argparse.ArgumentParser('RepeatNet')
    parser.add_argument("--model_name", type=str, default='wrn_16_4')
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--num_classes", type=int, default="10")
    parser.add_argument("--save_weights", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_activation", type=str, default='static_drop')
    parser.add_argument("--drop_rate", type=float, default=0.5)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    
    if not os.path.exists('logs'):
        os.mkdir('logs')

    if not os.path.exists('weights'):
        os.mkdir('weights')
    
    model = get_model(args.model_name, args.num_classes, args)
    system = CoolSystem(model, args.dataset)
    
    model_parameters = filter(lambda p: p.requires_grad, system.model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    log_name = args.model_name + '_' + args.dataset + '_params=' + str(int(params))
    checkpoint_callback = ModelCheckpoint(monitor='val_acc') if args.save_weights else False
    logger = loggers.TensorBoardLogger("logs", name=log_name, version=1)
    
    trainer = Trainer(default_root_dir='weights/', gpus=1, max_epochs=args.epochs, deterministic=True, gradient_clip_val=1, logger=logger, checkpoint_callback=args.save_weights)
    trainer.fit(system)
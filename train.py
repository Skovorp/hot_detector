import torch
import yaml
import json
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import HotDetectorDataset
from model import HotModel
from dotenv import load_dotenv
import wandb

load_dotenv()

with open('cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
random.seed(cfg['seed'])

wandb.init(
    name=cfg['run_name'],
    entity='sposiboh',
    project='hot_detector',
    config=cfg,
    mode='disabled' if cfg['run_name'] == 'debug' else 'online'
)

device = torch.device(cfg['device'])

train_dataset = HotDetectorDataset(partition='train', **cfg['data'])
val_dataset = HotDetectorDataset(partition='test', **cfg['data'])
print(val_dataset.get_stats())

train_loader = DataLoader(train_dataset, batch_size=cfg['training']['train_bs'], shuffle=True, num_workers=cfg['training']['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=cfg['training']['val_bs'], shuffle=False, num_workers=cfg['training']['num_workers'])

model = HotModel(**cfg['model']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

for e in range(cfg['training']['epochs']):
    for images, targets in train_loader:
        optimizer.zero_grad()
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss = (out - targets).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(loss.item())

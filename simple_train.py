import torch
import yaml
import json
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import HotDetectorPrecomputedDataset
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
from metrics import evaluate_ranking
from torch import nn
import torch
import torch.nn.functional as F

load_dotenv()

with open('cfg_simple.yaml', 'r') as f:
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

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = 4096
        self.stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d, 2 * d),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(2 * d, 1)
        )
        
    def forward(self, x):
        x = self.stack(x).squeeze(1)
        return x

device = torch.device(cfg['device'])

train_dataset = HotDetectorPrecomputedDataset(cfg['data']['train_path'])
val_dataset = HotDetectorPrecomputedDataset(cfg['data']['val_path'])

train_loader = DataLoader(train_dataset, batch_size=cfg['training']['train_bs'], shuffle=True, num_workers=cfg['training']['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=cfg['training']['val_bs'], shuffle=False, num_workers=cfg['training']['num_workers'])

model = SimpleModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

def listnet_loss(scores, p, tau=0.3, use_logit=True, label_smooth=0.0):
    with torch.no_grad():
        g = torch.log(p) - torch.log1p(-p) if use_logit else p
        q = torch.softmax(g / tau, dim=0)
        if label_smooth > 0:
            K = q.numel()
            q = (1 - label_smooth) * q + label_smooth / K
    qhat = torch.log_softmax(scores, dim=0)  # log prob
    return -(q * qhat).sum()

for e in range(cfg['training']['epochs']):
    model.train()
    correct_train_scores, predicted_train_scores = [], []
    pbar = tqdm(train_loader, desc=f'train {e+1}')
    for images, targets in pbar:
        optimizer.zero_grad()
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss = listnet_loss(out, targets, cfg['training']['tau'], cfg['training']['loss_use_logit'], cfg['training']['label_smooth'])
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
        correct_train_scores.extend(targets.float().detach().cpu().numpy().tolist())
        predicted_train_scores.extend(out.float().detach().cpu().numpy().tolist())
    train_metrics = evaluate_ranking(correct_train_scores, predicted_train_scores, 'train')
    print(train_metrics)
    wandb.log(train_metrics)
    
    model.eval()
    correct_val_scores, predicted_val_scores = [], []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'val {e+1}')
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            
            correct_val_scores.extend(targets.detach().cpu().numpy().tolist())
            predicted_val_scores.extend(predictions.detach().cpu().numpy().tolist())
    val_metrics = evaluate_ranking(correct_val_scores, predicted_val_scores, 'val')
    print(val_metrics)
    wandb.log(val_metrics)
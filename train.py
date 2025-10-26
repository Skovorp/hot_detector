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
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from metrics import evaluate_ranking

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

total_steps = len(train_loader) * cfg['training']['epochs']
warmup_steps = round(total_steps * cfg['training']['part_warmup'])
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

for e in range(cfg['training']['epochs']):
    model.train()
    pbar = tqdm(train_loader, desc=f'train {e+1}')
    for images, targets in pbar:
        optimizer.zero_grad()
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss = (out - targets).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        optimizer.step()
        lr_scheduler.step()
        pbar.set_postfix({"loss": loss.item()})
        wandb.log({
            'loss': loss.item(),
            'lr': lr_scheduler.get_last_lr()[0]
        })
    
    model.eval()
    correct_scores = []
    predicted_scores = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'val {e+1}')
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            
            correct_scores.extend(targets.detach().cpu().numpy().tolist())
            predicted_scores.extend(predictions.detach().cpu().numpy().tolist())
    val_metrics = evaluate_ranking(correct_scores, predicted_scores, 'val')
    print(val_metrics)
    wandb.log(val_metrics)
         
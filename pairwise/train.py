import torch
import yaml
import json
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import PairDatasetPreprocessed
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from models import 

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

train_dataset = PairDatasetPreprocessed(partition='train', **cfg['data'])
val_dataset = PairDatasetPreprocessed(partition='test', **cfg['data'])

train_loader = DataLoader(train_dataset, batch_size=cfg['training']['train_bs'], shuffle=True, num_workers=cfg['training']['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=cfg['training']['val_bs'], shuffle=False, num_workers=cfg['training']['num_workers'])

model_from = HotModel(**cfg['model']).to(device)
model_to = HotModel(**cfg['model']).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

total_steps = len(train_loader) * cfg['training']['epochs']
warmup_steps = round(total_steps * cfg['training']['part_warmup'])
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

for e in range(cfg['training']['epochs']):
    model.train()
    correct_train_scores, predicted_train_scores = [], []
    pbar = tqdm(train_loader, desc=f'train {e+1}')
    for images, targets in pbar:
        optimizer.zero_grad()
        images, targets = images.to(device), targets.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(images)
            loss = listnet_loss(out, targets, tau=cfg['training']['tau'], use_logit=cfg['training']['loss_use_logit'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        optimizer.step()
        lr_scheduler.step()
        pbar.set_postfix({"loss": loss.item()})
        wandb.log({
            'loss': loss.item(),
            'lr': lr_scheduler.get_last_lr()[0]
        })
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
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predictions = model(images)
            
            correct_val_scores.extend(targets.float().detach().cpu().numpy().tolist())
            predicted_val_scores.extend(predictions.float().detach().cpu().numpy().tolist())
    val_metrics = evaluate_ranking(correct_val_scores, predicted_val_scores, 'val')
    print(val_metrics)
    wandb.log(val_metrics)
model.save_finetune_weights()
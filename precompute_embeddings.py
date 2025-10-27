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
from transformers import AutoModel
import pickle

load_dotenv()

device = torch.device('cuda')

root_dir = "/root/hot_detector/exp_photos"

train_dataset = HotDetectorDataset(partition='train', root_dir=root_dir, resize_to=None)
val_dataset = HotDetectorDataset(partition='test', root_dir=root_dir, resize_to=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)

model = AutoModel.from_pretrained('facebook/dinov3-vit7b16-pretrain-lvd1689m').to(device)
model.eval()
pbar = tqdm(train_loader, desc=f'train')
all_image_embeddings, all_targets = [], []
for images, targets in pbar:
    with torch.no_grad():
        images, targets = images.to(device), targets.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(images).pooler_output
        all_image_embeddings.append(out.detach().cpu().float())
        all_targets.extend(targets)


train_embeddings = torch.cat(all_image_embeddings, dim=0).numpy()
train_targets = torch.tensor(all_targets).numpy()
with open('train_embs_no_rs.pickle', 'wb') as f:
    pickle.dump({'embeddings': train_embeddings, 'targets': train_targets}, f)

pbar = tqdm(val_loader, desc=f'val')
all_image_embeddings_val, all_targets_val = [], []
for images, targets in pbar:
    with torch.no_grad():
        images, targets = images.to(device), targets.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(images).pooler_output
        all_image_embeddings_val.append(out.detach().cpu().float())
        all_targets_val.extend(targets)

val_embeddings = torch.cat(all_image_embeddings_val, dim=0).numpy()
val_targets = torch.tensor(all_targets_val).numpy()
with open('val_embs_no_rs.pickle', 'wb') as f:
    pickle.dump({'embeddings': val_embeddings, 'targets': val_targets}, f)
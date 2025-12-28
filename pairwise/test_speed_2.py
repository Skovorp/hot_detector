import time
from dataset import PairDatasetPreprocessed
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def make_ds():
    return PairDatasetPreprocessed(partition='train', cache_dir='/root/hot_detector/pairwise/image_cache', num_items_epoch=10000)

if __name__ == '__main__':
    batch_size = 64
    
    device = torch.device('cuda')
    
    ds = make_ds()
    loader_collate = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True, pin_memory=True)
    st = time.time()
    for b in tqdm(loader_collate):
        _, _, _ = b[0].to(device), b[1].to(device), b[2].to(device)
    passed = time.time() - st
    print(f'DataLoader (with collate). Time taken: {passed:.4f} seconds. Batches per second: {len(loader_collate) / passed:.2f}')
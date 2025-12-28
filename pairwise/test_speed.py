import time
from dataset import PairDatasetPreprocessed
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def make_ds():
    return PairDatasetPreprocessed(partition='train', cache_dir='/root/hot_detector/pairwise/image_cache', num_items_epoch=64 * 10)

if __name__ == '__main__':
    batch_size = 64
    
    # 1. Custom loop - no collation (just list of tuples)
    ds = make_ds()
    idxs = torch.randperm(len(ds))
    batches = [idxs[i:i+batch_size] for i in range(0, len(ds) // batch_size * batch_size, batch_size)]
    print(f"Number of batches: {len(batches)}")
    st = time.time()
    for ind in tqdm(batches):
        b = [ds[i] for i in ind]
    print(f'1. Custom (no collate). Time taken: {time.time() - st:.4f} seconds')
    del ds

    # 2. Custom loop WITH collation (stack tensors like DataLoader does)
    ds = make_ds()
    idxs = torch.randperm(len(ds))
    batches = [idxs[i:i+batch_size] for i in range(0, len(ds) // batch_size * batch_size, batch_size)]
    st = time.time()
    for ind in tqdm(batches):
        items = [ds[i] for i in ind]
        b = (torch.stack([x[0] for x in items]), 
             torch.stack([x[1] for x in items]), 
             torch.stack([x[2] for x in items]))
    print(f'2. Custom (with collate). Time taken: {time.time() - st:.4f} seconds')
    del ds

    # 3. DataLoader WITHOUT collation
    ds = make_ds()
    loader_no_collate = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=lambda x: x)
    st = time.time()
    for b in tqdm(loader_no_collate):
        pass
    print(f'3. DataLoader (no collate). Time taken: {time.time() - st:.4f} seconds')
    del ds

    # 4. DataLoader WITH default collation
    ds = make_ds()
    loader_collate = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    st = time.time()
    for b in tqdm(loader_collate):
        pass
    print(f'4. DataLoader (with collate). Time taken: {time.time() - st:.4f} seconds')
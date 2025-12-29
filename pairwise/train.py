import itertools
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb
import pandas as pd
from safetensors.torch import safe_open
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from metrics import evaluate_ranking

from dataset import PairDatasetPreprocessed
from models import Combine, EmbedModel

# mp.set_sharing_strategy("file_system")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_weights(output_dir: Path, cfg: Dict, from_model: EmbedModel, to_model: EmbedModel, combiner: Combine) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": from_model.state_dict(), "config": cfg},
        output_dir / "from_model.pt",
    )
    torch.save(
        {"state_dict": to_model.state_dict(), "config": cfg},
        output_dir / "to_model.pt",
    )
    torch.save(
        {"state_dict": combiner.state_dict(), "config": cfg},
        output_dir / "combiner.pt",
    )


def forward_pair(
    from_model: EmbedModel,
    to_model: EmbedModel,
    combiner: Combine,
    from_batch: torch.Tensor,
    to_batch: torch.Tensor,
) -> torch.Tensor:
    from_embed = from_model(from_batch)
    to_embed = to_model(to_batch)
    concat = torch.cat((from_embed, to_embed), dim=1)
    logits = combiner(concat).squeeze(-1)
    return logits


def shutdown_loader(loader: DataLoader) -> None:
    """
    Forcefully tear down DataLoader worker pools so their shared-memory
    allocations are released before spinning up the next set of workers.
    This helps avoid hitting /dev/shm limits when alternating between large
    train/val loaders in the same process.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    loader._iterator = None  # type: ignore[attr-defined]
    
def preload_absolute_eval_data(cache_dir: Path, profiles_path: Path, device: torch.device):
    """Preload all data needed for absolute evaluation."""
    df = pd.read_parquet(profiles_path)[["photo_path", "sex", "part_likes"]]
    df["stem"] = df["photo_path"].apply(lambda x: Path(x).stem)
    
    mapping = json.load(open(cache_dir / "partition_mapping.json"))
    partitions = {p: safe_open(p, framework="pt", device="cpu") for p in set(mapping.values())}
    
    males = df[df["sex"] == "male"].reset_index(drop=True)
    females = df[df["sex"] == "female"].reset_index(drop=True)
    
    def load_enc(stem):
        return partitions[mapping[stem]].get_tensor(stem)
    
    male_enc = torch.stack([load_enc(s) for s in males["stem"]]).to(device)
    female_enc = torch.stack([load_enc(s) for s in females["stem"]]).to(device)
    
    return males, females, male_enc, female_enc


def evaluate_absolute(
    from_model: EmbedModel,
    to_model: EmbedModel,
    combiner: Combine,
    males: pd.DataFrame,
    females: pd.DataFrame,
    male_enc: torch.Tensor,
    female_enc: torch.Tensor,
) -> Dict:
    """Evaluate absolute popularity prediction."""
    from_model.eval()
    to_model.eval()
    combiner.eval()
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        male_from = from_model(male_enc)
        male_to = to_model(male_enc)
        female_from = from_model(female_enc)
        female_to = to_model(female_enc)
    
    # score males: each male as "to", all females as "from"
    male_scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(males)):
            concat = torch.cat([female_from, male_to[i:i+1].expand(len(females), -1)], dim=1)
            logits = combiner(concat).squeeze(-1)
            male_scores.append(torch.sigmoid(logits).mean().item())
    
    # score females: each female as "to", all males as "from"
    female_scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(females)):
            concat = torch.cat([male_from, female_to[i:i+1].expand(len(males), -1)], dim=1)
            logits = combiner(concat).squeeze(-1)
            female_scores.append(torch.sigmoid(logits).mean().item())
    
    male_metrics = evaluate_ranking(males["part_likes"].tolist(), male_scores, "male")
    female_metrics = evaluate_ranking(females["part_likes"].tolist(), female_scores, "female")
    
    return {'avg_tau': (male_metrics['male_kendall_tau'] + female_metrics['female_kendall_tau']) / 2, **male_metrics, **female_metrics}


def main(cfg_path) -> None:
    load_dotenv()

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    save_dir = Path(cfg["training"]["output_dir"]).expanduser().resolve() / cfg["run_name"]
        
    wandb.init(
        name=cfg['run_name'],
        entity='sposiboh',
        project='hot_detector',
        config=cfg,
        mode='disabled' if cfg['run_name'] == 'debug' else 'online'
    )

    set_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    train_dataset = PairDatasetPreprocessed(partition="train", **cfg["data"])
    val_dataset = PairDatasetPreprocessed(partition="test", **cfg["data"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["train_bs"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        prefetch_factor=cfg["training"]["prefetch_factor"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["val_bs"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        prefetch_factor=cfg["training"]["prefetch_factor"]
    )

    from_model = EmbedModel(cfg["model"]["use_layers"], cfg["model"]["neftune_alpha"]).to(device)
    to_model = EmbedModel(cfg["model"]["use_layers"], cfg["model"]["neftune_alpha"]).to(device)
    combiner = Combine().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer_from = torch.optim.AdamW(
        from_model.parameters(),
        lr=cfg["training"]["lr_embed"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    optimizer_to = torch.optim.AdamW(
        to_model.parameters(),
        lr=cfg["training"]["lr_embed"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    optimizer_comb = torch.optim.AdamW(
        combiner.parameters(),
        lr=cfg["training"]["lr_combiner"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = max(1, len(train_loader) * cfg["training"]["epochs"])
    warmup_steps = max(1, int(total_steps * cfg["training"]["part_warmup"]))

    scheduler_from = get_cosine_schedule_with_warmup(
        optimizer_from, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scheduler_to = get_cosine_schedule_with_warmup(
        optimizer_to, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    # scheduler_comb = get_cosine_schedule_with_warmup(
    #     optimizer_comb, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    # )
    
     # Preload absolute eval data
    cache_dir = Path(cfg["data"]["cache_dir"])
    profiles_path = cache_dir / "_test_profiles.parquet"
    males, females, male_enc, female_enc = preload_absolute_eval_data(cache_dir, profiles_path, device)
    print(f"Singleplayer ranking eval: {len(males)} males, {len(females)} females")

    for epoch in range(cfg["training"]["epochs"]):
        from_model.train()
        to_model.train()
        combiner.train()

        train_loss = 0.0
        train_targets = []
        train_scores = []

        pbar = tqdm(train_loader, desc=f"train {epoch + 1}")
        for from_batch, to_batch, targets in pbar:
            from_batch = from_batch.to(device, non_blocking=True)
            to_batch = to_batch.to(device, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32)

            optimizer_from.zero_grad(set_to_none=True)
            optimizer_to.zero_grad(set_to_none=True)
            optimizer_comb.zero_grad(set_to_none=True)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = forward_pair(from_model, to_model, combiner, from_batch, to_batch)
                loss = criterion(logits, targets)

            loss.backward()

            if cfg["training"]["grad_clip"] is not None:

                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        from_model.parameters(),
                        to_model.parameters(),
                        combiner.parameters(),
                    ),
                    cfg["training"]["grad_clip"],
                )

            optimizer_from.step()
            optimizer_to.step()
            optimizer_comb.step()

            scheduler_from.step()
            scheduler_to.step()
            # scheduler_comb.step()

            probs = torch.sigmoid(logits).detach().cpu().tolist()
            train_scores.extend(probs)
            train_targets.extend(targets.detach().cpu().tolist())
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({
                'loss': loss.item(),
                'lr_from': scheduler_from.get_last_lr()[0],
                'lr_to': scheduler_to.get_last_lr()[0],
                # 'lr_comb': scheduler_comb.get_last_lr()[0],
            })

        train_loss /= max(1, len(train_loader))
        train_ap = average_precision_score(train_targets, train_scores)
        wandb.log({'train_loss': train_loss, 'train_ap': train_ap})
        print({'train_loss': train_loss, 'train_ap': train_ap})

        shutdown_loader(train_loader)

        from_model.eval()
        to_model.eval()
        combiner.eval()

        val_loss = 0.0
        val_targets = []
        val_scores = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"val {epoch + 1}")
            for from_batch, to_batch, targets in pbar:
                from_batch = from_batch.to(device, non_blocking=True)
                to_batch = to_batch.to(device, non_blocking=True)
                targets = targets.to(device=device, dtype=torch.float32)

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = forward_pair(from_model, to_model, combiner, from_batch, to_batch)
                    loss = criterion(logits, targets)

                probs = torch.sigmoid(logits).detach().cpu().tolist()
                val_scores.extend(probs)
                val_targets.extend(targets.detach().cpu().tolist())
                val_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        shutdown_loader(val_loader)

        val_loss /= max(1, len(val_loader))
        val_ap = average_precision_score(val_targets, val_scores)
        wandb.log({'val_loss': val_loss, 'val_ap': val_ap})
        print({'val_loss': val_loss, 'val_ap': val_ap})
        
        # Absolute evaluation
        abs_metrics = evaluate_absolute(
            from_model, to_model, combiner, males, females, male_enc, female_enc
        )
        wandb.log(abs_metrics)
        print(abs_metrics)

        print(
            f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
            f"train_loss: {train_loss:.4f} | train_AP: {train_ap:.4f} | "
            f"val_loss: {val_loss:.4f} | val_AP: {val_ap:.4f}"
        )
        train_dataset.resample_data()
        save_weights(save_dir, cfg, from_model, to_model, combiner)
        print(f"Saved weights to {save_dir}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "cfg.yaml"
    main(cfg_path)
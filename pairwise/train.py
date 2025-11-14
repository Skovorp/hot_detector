import itertools
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb

from dataset import PairDatasetPreprocessed
from models import Combine, EmbedModel


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


def main() -> None:
    load_dotenv()

    cfg_path = Path(__file__).with_name("cfg.yaml")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
        
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
        # pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["val_bs"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        # pin_memory=device.type == "cuda",
    )

    from_model = EmbedModel(cfg["model"]["use_layers"]).to(device)
    to_model = EmbedModel(cfg["model"]["use_layers"]).to(device)
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

            probs = torch.sigmoid(logits).detach().cpu().tolist()
            train_scores.extend(probs)
            train_targets.extend(targets.detach().cpu().tolist())
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({
                'loss': loss.item(),
                'lr_from': scheduler_from.get_last_lr()[0],
                'lr_to': scheduler_to.get_last_lr()[0],
            })

        train_loss /= max(1, len(train_loader))
        train_ap = average_precision_score(train_targets, train_scores)
        wandb.log({'train_loss': train_loss, 'train_ap': train_ap})
        print({'train_loss': train_loss, 'train_ap': train_ap})

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

        val_loss /= max(1, len(val_loader))
        val_ap = average_precision_score(val_targets, val_scores)
        wandb.log({'val_loss': val_loss, 'val_ap': val_ap})
        print({'val_loss': val_loss, 'val_ap': val_ap})

        print(
            f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
            f"train_loss: {train_loss:.4f} | train_AP: {train_ap:.4f} | "
            f"val_loss: {val_loss:.4f} | val_AP: {val_ap:.4f}"
        )

    save_dir = (
        Path(cfg["training"]["output_dir"]).expanduser().resolve() / cfg["run_name"]
    )
    save_weights(save_dir, cfg, from_model, to_model, combiner)
    print(f"Saved weights to {save_dir}")


if __name__ == "__main__":
    main()
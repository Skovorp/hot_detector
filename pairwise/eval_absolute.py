"""Evaluate pairwise model's ability to predict absolute popularity scores."""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from metrics import evaluate_ranking
from pairwise.models import EmbedModel, Combine

CACHE_DIR = Path("/root/hot_detector/pairwise/image_cache")
CKPT_DIR = Path("/root/hot_detector/pairwise/checkpoints/50k_1layer_high_wd_05")
PROFILES_PATH = Path("/root/used_images/_test_profiles.parquet")


def load_models(device):
    from_model = EmbedModel(1).to(device)
    to_model = EmbedModel(1).to(device)
    combiner = Combine().to(device)
    from_model.load_state_dict(torch.load(CKPT_DIR / "from_model.pt")["state_dict"])
    to_model.load_state_dict(torch.load(CKPT_DIR / "to_model.pt")["state_dict"])
    combiner.load_state_dict(torch.load(CKPT_DIR / "combiner.pt")["state_dict"])
    from_model.eval(); to_model.eval(); combiner.eval()
    return from_model, to_model, combiner


def main():
    device = torch.device("cuda")
    df = pd.read_parquet(PROFILES_PATH)[["photo_path", "sex", "part_likes"]]
    df["stem"] = df["photo_path"].apply(lambda x: Path(x).stem)
    
    mapping = json.load(open(CACHE_DIR / "partition_mapping.json"))
    partitions = {p: safe_open(p, framework="pt", device="cpu") for p in set(mapping.values())}
    
    # filter to available stems
    df = df[df["stem"].isin(mapping.keys())].reset_index(drop=True)
    males = df[df["sex"] == "male"].reset_index(drop=True)
    females = df[df["sex"] == "female"].reset_index(drop=True)
    print(f"Males: {len(males)}, Females: {len(females)}")
    
    # load all encodings
    def load_enc(stem):
        return partitions[mapping[stem]].get_tensor(stem)
    
    male_enc = torch.stack([load_enc(s) for s in males["stem"]]).to(device)
    female_enc = torch.stack([load_enc(s) for s in females["stem"]]).to(device)
    
    from_model, to_model, combiner = load_models(device)
    
    # embed all
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        male_from = from_model(male_enc)
        male_to = to_model(male_enc)
        female_from = from_model(female_enc)
        female_to = to_model(female_enc)
    
    # score males: each male as "to" (features), all females as "from" (preferences)
    male_scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in tqdm(range(len(males)), desc="Scoring males"):
            concat = torch.cat([female_from, male_to[i:i+1].expand(len(females), -1)], dim=1)
            logits = combiner(concat).squeeze(-1)
            male_scores.append(torch.sigmoid(logits).mean().item())
    
    # score females: each female as "to", all males as "from"  
    female_scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in tqdm(range(len(females)), desc="Scoring females"):
            concat = torch.cat([male_from, female_to[i:i+1].expand(len(males), -1)], dim=1)
            logits = combiner(concat).squeeze(-1)
            female_scores.append(torch.sigmoid(logits).mean().item())
    
    # evaluate
    print("\n=== Male Ranking ===")
    male_metrics = evaluate_ranking(males["part_likes"].tolist(), male_scores, "male")
    for k, v in male_metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Female Ranking ===")
    female_metrics = evaluate_ranking(females["part_likes"].tolist(), female_scores, "female")
    for k, v in female_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()


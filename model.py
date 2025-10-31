from torch import nn
from transformers import AutoModel
import os
import torch
from datetime import datetime

class HotModel(nn.Module):
    def __init__(self, backbone, bb_dim, freeze_bb) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.proj = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(bb_dim, 2 * bb_dim),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(2 * bb_dim, 1)
        )
        if freeze_bb:
            for param in self.backbone.parameters():
                param.requires_grad = False
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        for param in self.backbone.layer[-1].parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.backbone(x).pooler_output
        x = self.proj(x).squeeze(1)
        return x

    def save_finetune_weights(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"ft_weights_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.proj.state_dict(), os.path.join(folder_name, "proj.pt"))
        torch.save(self.backbone.layer[-1].state_dict(), os.path.join(folder_name, "layer_-1.pt"))
        torch.save(self.backbone.norm.state_dict(), os.path.join(folder_name, "norm.pt"))
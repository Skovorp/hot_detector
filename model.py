from torch import nn
from transformers import AutoModel

class HotModel(nn.Module):
    def __init__(self, backbone, bb_dim) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.proj = nn.Linear(bb_dim, 1)
        
    def forward(self, x):
        x = self.backbone(x).pooler_output
        x = self.proj(x).squeeze(1)
        return x

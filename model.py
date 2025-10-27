from torch import nn
from transformers import AutoModel

class HotModel(nn.Module):
    def __init__(self, backbone, bb_dim, freeze_bb) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.proj = nn.Linear(bb_dim, 1)
        if freeze_bb:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.backbone(x).pooler_output
        x = self.proj(x).squeeze(1)
        return x

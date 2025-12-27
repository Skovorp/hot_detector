from torch import nn
import torch
from transformers import AutoModel
from dotenv import load_dotenv

load_dotenv()

class PreprocModel(nn.Module):
    def __init__(self, skip_layers):
        super().__init__()
        full_model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
        self.embeddings = full_model.embeddings
        self.rope_embeddings = full_model.rope_embeddings
        self.encoder_layers = nn.ModuleList(full_model.layer[:-skip_layers]) 
        del full_model
    
    def forward(self, x):
        position_embeddings = self.rope_embeddings(x)
        x = self.embeddings(x)
        for layer in self.encoder_layers:
            x = layer(x, position_embeddings=position_embeddings)
        return x

class EmbedModel(nn.Module):
    def __init__(self, use_layers, part_mask=None):
        super().__init__()
        full_model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
        self.rope_embeddings = full_model.rope_embeddings
        self.encoder_layers = nn.ModuleList(full_model.layer[-use_layers:]) 
        self.norm = full_model.norm
        self.part_mask = part_mask
        self.mask_token = nn.Parameter(torch.randn(1, 1, 4096))
        del full_model
        
    def forward(self, x):
        if self.part_mask and self.training:
            mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < self.part_mask
            mask[:, :5, :] = False  # don't mask CLS + 4 register tokens
            x = torch.where(mask, self.mask_token, x)
        position_embeddings = self.rope_embeddings(torch.empty(1, 3, 512, 512, device=x.device, dtype=x.dtype))
        for layer in self.encoder_layers:
            x = layer(x, position_embeddings=position_embeddings)
        x = self.norm(x)
        x = x[:, 0, :]
        return x
    
class Combine(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096 * 2, 4096),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(4096, 1)
        )
    
    def forward(self, x):
        x = self.stack(x)
        return x
    
def test():
    device = torch.device('cuda')
    prep = PreprocModel(2).to(device)
    embed = EmbedModel(2).to(device)
    prep.eval()
    embed.eval()
    inp = torch.rand(1, 3, 512, 512).to(device)
    out = embed(prep(inp))
    
    real_model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m").to(device)
    real_model.eval()
    res = real_model(pixel_values=inp).pooler_output
    assert (res - out).abs().max().item() == 0.0
    
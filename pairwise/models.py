from torch import nn
import torch
from transformers import AutoModel
from dotenv import load_dotenv

load_dotenv()

def neftune(x, alpha):
    assert len(x.shape) == 3, "need B, L, d tensor"
    noise = torch.rand(x.shape, device=x.device, dtype=x.dtype) * 2 - 1
    scale = alpha / (x.size(1) * x.size(2)) ** 0.5
    return x + noise * scale

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
    def __init__(self, use_layers, neftune_alpha):
        super().__init__()
        full_model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
        self.rope_embeddings = full_model.rope_embeddings
        self.encoder_layers = nn.ModuleList(full_model.layer[-use_layers:]) 
        self.norm = full_model.norm
        del full_model
        self.neftune_alpha = neftune_alpha
        
    def forward(self, x):
        position_embeddings = self.rope_embeddings(torch.empty(1, 3, 512, 512, device=x.device, dtype=x.dtype))
        if self.neftune_alpha is not None and self.training:
            x = neftune(x, self.neftune_alpha)
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
            nn.Linear(4096 * 2, 256),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(256, 1)
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
    
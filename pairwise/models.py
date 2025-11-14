from torch import nn
import torch

class PreprocModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.transformer_layers = None # everything except for last 2 layers of AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
    
    def forward(self, x):
        x = self.transformer_layers(x)
        return x

class EmbedModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.transformer_layers = None # last 2 layers of AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m") and pooler
        
    def forward(self, x):
        x = self.transformer_layers(x)
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
    
if __name__ == "__main__":
    from random import random
    from time import time
    ina = [random() for _ in range(4096)]
    inb = [random() for _ in range(4096)]

    st = time()
    with torch.no_grad():
        m = Combine()
        m.load_state_dict(torch.load("combine_weights.pth"))
        a = torch.tensor(ina).unsqueeze(0)
        b = torch.tensor(inb).unsqueeze(0)
        
        inp = torch.cat([a, b], 1)
        print(torch.sigmoid(m(inp)).item())
    print(time() - st)
    
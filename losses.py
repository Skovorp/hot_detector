import torch

def listnet_loss(scores, p, tau=0.3, use_logit=True, label_smooth=0.0):
    with torch.no_grad():
        g = torch.log(p) - torch.log1p(-p) if use_logit else p
        q = torch.softmax(g / tau, dim=0)
        if label_smooth > 0:
            K = q.numel()
            q = (1 - label_smooth) * q + label_smooth / K
    qhat = torch.log_softmax(scores, dim=0)  # log prob
    return -(q * qhat).sum()
import torch

def tv_loss_2d(y_pred, reduction='mean'):
    """2D总变差损失"""
    diff_h = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :])
    diff_v = torch.abs(y_pred[..., :, 1:] - y_pred[..., :, :-1])
    if reduction == 'mean':
        return (diff_h.mean() + diff_v.mean()) / 2
    else:
        return (diff_h.sum() + diff_v.sum()) / 2
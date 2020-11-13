import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 


# ohem loss function
def ohem_loss_function(logits, targets, rate):
    batch_size = logits.size(0)
    loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=-1)
    keep_num = min(batch_size, int(batch_size * rate))
    ohem_cls_loss, _ = loss.topk(keep_num)
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


# label smooth
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, eps=0.1, reduction="mean"):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.bias = 1e-7
#         self.eps = eps 
#         self.reduction = reduction

#     def forward(self, output, target):
#         c = output.size()[-1]
#         log_preds = F.log_softmax(output + self.bias, dim=-1) 
#         if self.reduction == 'sum':
#             loss = -log_preds.sum()
#         else:
#             loss = -log_preds.sum(dim=-1) + self.bias
#             if self.reduction == 'mean':
#                 loss = loss.mean()
#         return loss * self.eps / c + (1 - self.eps) * F.cross_entropy(output, target, reduction=self.reduction)


# label Smooth from github https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, eps=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - eps
        self.smoothing = eps
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# from https://github.com/rwightman/pytorch-image-models/blob/7613094fb5cb960813f606a5c42e3c00c961bc8f/timm/loss/cross_entropy.py
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# from https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        x = x.float()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

# DRW_weights 
def drw_weights(epoch, cls_num_list, interval=30):
    idx = epoch // interval
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return per_cls_weights


if __name__ == "__main__":
    a = torch.randn(2, 2)
    b = torch.randn(2, ).long()
    loss = LabelSmoothingCrossEntropy()
    output = loss(a, b)
    print(output)



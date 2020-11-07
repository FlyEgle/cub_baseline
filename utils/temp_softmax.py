"""
Soft label for Crossentropy loss 
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class SoftCrossentropyLoss(nn.Module):
    r"""Apply softtarget for crossentropy

    Arguments:
        temperateure (int): a scale for the soft target
        reduction (str): "mean" for the mean loss with the target*logits
    """
    def __init__(self, temperature=5, reduction="mean"):
        super(SoftCrossentropyLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-7
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits):
        t_prob = F.softmax(teacher_logits / self.temperature, 1) + self.eps 
        s_log_prob = -F.log_softmax(student_logits / self.temperature, 1) + self.eps
        batch = student_logits.shape[0]
        if self.reduction == "mean":
            loss = torch.sum(torch.mul(s_log_prob, t_prob)) / batch 
        else:
            loss = torch.sum(torch.mul(s_log_prob, t_prob))
        return loss


class KLSoftLoss(nn.Module):
    r"""Apply softtarget for kl loss

    Arguments:
        reduction (str): "batchmean" for the mean loss with the p(x)*(log(p(x)) - log(q(x)))
    """
    def __init__(self, temperature=1, reduction="batchmean"):
        super(KLSoftLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-7
        self.temperature = temperature
        self.klloss = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, s_logits, t_logits):
        s_prob = F.log_softmax(s_logits/self.temperature, 1)
        t_prob = F.softmax(t_logits/self.temperature, 1) 
        loss = self.klloss(s_prob, t_prob) * self.temperature * self.temperature
        return loss


if __name__ == "__main__":
    student = torch.tensor([[1,2,3]], dtype=torch.float)
    teacher = torch.tensor([[2,3,4]], dtype=torch.float)
    loss = SoftCrossentropyLoss()
    output = loss(student, teacher)
    print(output)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-7

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(
                logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros(
            [batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class FocalLossV1(nn.Module):  # 1d and 2d

    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLossV1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1]*2  # [0.5, 0.5]

            prob = torch.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            #logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            # prob = F.log_softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            # onehot->label
            select.scatter_(1, target, 1.)

        prob = prob*select
        prob = torch.clamp(prob, 1e-8, 1-1e-8)
        batch_loss = -1 * self.alpha * (1-prob)**self.gamma * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


class EasyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(EasyFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
    
    def forward(self, logits, targets, class_weights=None):
        ce_loss = F.cross_entropy(logits, targets)
        pt = torch.exp(-ce_loss) + 1e-7
        focal_loss = (self.alpha*(1-pt)**self.gamma * ce_loss).mean()
        return focal_loss


if __name__ == "__main__":
    focal_loss = EasyFocalLoss()
    for i in range(1000):
        data_tensor = torch.randn(2, 10).float().cuda()
        data_lbl = torch.randn(2, ).long().cuda()
        output_loss = focal_loss(data_tensor, data_lbl)
        print("output_loss", output_loss)

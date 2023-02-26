import torch.nn as nn
import torch
import numpy as np

class MarginLoss(nn.Module):

    def __init__(self, margin=1.0, target=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.target = target

    def forward(self, logits, label):
        if not self.target:
            one_hot = torch.zeros_like(logits, dtype=torch.bool)
            label = label.reshape(-1, 1)
            one_hot.scatter_(1, label, 1)
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]# 64
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label], logits[:, (label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin

        return margin.mean()

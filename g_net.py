import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 512

class GNet (nn.Module):
    def __init__ (self):
        super(GNet, self).__init__()
        self.linear1a = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.linear1b = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        # self.scaling = Scaling(EMBEDDING_DIM, method="unit")

    def forward (self, X1, X2):
        linear = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear
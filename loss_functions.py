import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, g_net, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)
        self.g_net = g_net
        self.eps = eps
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embedding: torch.Tensor, ground_truth):
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.W))
        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(cos_theta.transpose(0, 1)[ground_truth]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((cos_theta[i, :y], cos_theta[i, y+1:])).unsqueeze(0) for i, y in enumerate(ground_truth)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

    def getRes(self, x):
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        res = torch.argmax(wf, 1)
        return res

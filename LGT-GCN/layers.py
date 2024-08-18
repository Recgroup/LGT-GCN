import torch
import torch.nn as nn
from  data import  *
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, args,hidden_size):
        super(LinearAttention, self).__init__()
        self.args = args
        self.hid = hidden_size
        if args.data in ['cora','citeseer','pubmed']:
            self.att = nn.Sequential(nn.Linear(self.hid,self.hid),nn.ReLU())
        else:
            self.att = nn.Sequential(nn.Linear(2*self.hid,1),nn.ReLU())

    def forward(self, Q, K):   # h 2708XcX64
        if self.args.data in ['cora', 'citeseer', 'pubmed']:
            Q = self.att(Q)
            K = self.att(K)
            attention = F.gumbel_softmax(F.dropout(torch.matmul(Q, K.permute(0, 2, 1)),0.1).squeeze(dim=1))
        else:
            Q = Q.repeat(1, K.shape[1], 1)
            attention = torch.softmax(F.dropout(self.att(torch.cat([Q, K], dim=-1)).squeeze(-1),0.2), dim=-1)
        return attention


class Local_Layer(nn.Module):
    def __init__(self, args,hid,head_num,nclass):
        super(Local_Layer, self).__init__()

        self.hid = hid
        self.nclass = nclass
        self.head_dim = hid // head_num
        self.head_num = head_num

        self.attentionLayers = nn.ModuleList([LinearAttention(args,self.head_dim) for _ in range(self.head_num)])

    def  forward(self,h,s):
        s = s.view(s.shape[0],-1,self.head_num,self.head_dim)
        h = h.view(h.shape[0],-1,self.head_num,self.head_dim)#.expand_as(s)
        attentions = [attentionLayer(h[:,:,idx,:],s[:,:,idx,:]).unsqueeze(1) for idx,attentionLayer  in enumerate(self.attentionLayers)]
        h = torch.cat([torch.matmul(attention,s[:,:,idx,:]) for idx,attention in enumerate(attentions)],dim=-1)
        return h.squeeze(1)

class Globel_Layer(nn.Module):
    def __init__(self,dropout):
        super(Globel_Layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def get_G_information(self,h,G):
        n = h.shape[0]
        matrix01 = self.dropout(torch.mm(h,G.T))
        att = F.gumbel_softmax(matrix01, tau=0.5, dim=-1)
        H = torch.mm(att, G)
        return H
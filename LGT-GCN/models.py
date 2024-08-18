from layers import *
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class LGT_GCN(nn.Module):
    def __init__(self, args, nfeat, nclass):
        super(LGT_GCN, self).__init__()

        self.args =args
        self.nclass = nclass
        self.hid  = args.hid
        self.layers = args.nlayer
        self.tau = args.tau
        self.head_num = 1
        self.dropoutRate = args.dropout
        self.smooth_num = args.smooth_num
        self.maskingRate = args.maskingRate

        self.C = int(args.C*self.nclass)
        self.iter = 0
        self.alpha = args.alpha # progressive layer staking 1
        self.beta =args.beta # open global  and get global proportion 0.1
        self.local =  True if args.local >0 else False
        self.iter_num = args.epochs

        #drop
        self.dropout = nn.Dropout(p=self.dropoutRate).train()
        self.fc = torch.nn.Linear(nfeat,self.hid)
        self.classify = torch.nn.Linear(self.hid,self.nclass)

        if self.local:
            #local
            self.local_Layer = Local_Layer(self.args,self.hid,self.head_num,self.nclass)

        if self.beta > 0:
            # global
            self.init_G = False
            self.G = torch.nn.Parameter(torch.randn(self.C, self.hid))
            self.P = torch.nn.Sequential(
                    torch.nn.Linear(self.hid, self.hid), torch.nn.ReLU(),
                    torch.nn.Linear(self.hid, self.C), torch.nn.ReLU())
            self.globe_Layer = Globel_Layer(self.dropoutRate)

    def uncorrelationloss(self,z):
        I = torch.eye(z.shape[1]).to(z.device)
        H = torch.ones_like(I)-I
        m = torch.mm(torch.mm(z,H),z.T)
        I2 = torch.eye(z.shape[0]).to(z.device)
        loss = ((m-I2)**2).mean()
        return loss

    def masking(self, input, mask_prob=0.3):
        random_mask = torch.empty(
            (input.shape[0], input.shape[1]),
            dtype=torch.float32,
            device=input.device).uniform_(0, 1) > mask_prob
        return random_mask * input

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastiveloss(self, z1,z2,adj):
        z1 = self.masking(z1,self.maskingRate)
        z2 = self.masking(z2,self.maskingRate)
        adj_mask = torch.where(adj.to_dense()>0,torch.ones(1).to(adj.device),torch.zeros(1).to(adj.device))
        adj_mask.fill_diagonal_(1)
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        CT = -torch.log((between_sim*adj_mask).sum(1) / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1) - (between_sim*adj_mask).sum(1)))
        return CT.mean()

    def smooth(self, x, adj, num):
        for i in range(num):
            x = adj.mm(x)
        return x

    def generate_list(self, input_list):
        output_list = [input_list[0]]
        for i in range(1, len(input_list)):
            output_list.append(output_list[i - 1] + input_list[i])
        return output_list

    def find_first_greater_than_k(self, input_list, k):
        try:
            index = next(i for i, value in enumerate(input_list) if value > k)
            return index
        except StopIteration:
            return len(input_list) - 1

    def forward(self, input, adj):
        input = self.dropout(input)
        h = self.fc(input)

        #compute
        h0 = h
        s = h
        loss = 0
        if self.alpha > 0:
            a = np.array([self.alpha ** i for i in range(self.layers, 0, -1)])
            a = a / a.sum() * self.iter_num
            sum_list = self.generate_list(a)
            flag = self.find_first_greater_than_k(sum_list, self.iter)

        if self.beta > 0:
            if not self.init_G:
                self.init_G = True
                # PCA and Kmeans init
                data = input.cpu().detach().numpy()
                pca = PCA(n_components=self.hid)
                data = pca.fit_transform(data)
                kmeans = KMeans(n_clusters=self.C)
                kmeans.fit(data)
                init_node = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(h.device)
                self.G.data = init_node

        for idx in range(self.layers):

            if self.alpha > 0:
                if idx<= flag:
                    pass
                    if self.training:
                        print(f'layer:{idx}, current epochs:{self.iter}')
                else:
                    continue

            if self.beta > 0 and self.init_G:
                h_g = self.globe_Layer.get_G_information(h.detach(), self.G.detach())
                h_g = self.dropout(h_g)
                h = h * (1 - self.beta) + h_g * self.beta

            if self.local:
                s = torch.cat([adj.mm(h),s],dim=-1)
                h = self.dropout(self.local_Layer((h+h0)/2,s))
            else:
                h = adj.mm(h)
        # contrastive loss
        loss += self.layers*self.contrastiveloss(h, self.smooth(h0, adj, self.smooth_num), adj)

        if self.beta > 0:
            p0 = self.P(h.detach())
            p = F.normalize(p0, dim=-1, p=1)
            loss += (torch.norm(h.detach() - torch.mm(p, self.G.detach())) + torch.norm(
                h.detach() - torch.mm(p.detach(), self.G)))
            loss += torch.norm(torch.norm(p0, p=1, dim=-1), p=2)
            loss += self.uncorrelationloss(self.G)


        # classifing
        h = self.dropout(h)
        y = torch.softmax(self.classify(h),dim=-1) if self.args.data in ['citeseer','cora','pubmed'] else self.classify(h)
        if self.training:
            self.iter += 1
        return y,loss,0,0


import argparse
import models
from utils import train, val_and_test
from data import load_data
import torch
import  random
import os
import time
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#parameter
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='citeseer', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='LGT_GCN', help='{LGT_GCN}')
parser.add_argument('--hid', type=int, default= 128, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--wightDecay', type=float, default=5e-4, help='Weight decay .')
parser.add_argument('--nlayer', type=int, default= 8,  help='Number of layers, works for Deep model.')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or dont use cuda')
parser.add_argument("--seed",type=int,default=30,help="seed for model")

parser.add_argument("--alpha",type=int,default=1, help="alpha for LGT in model")
parser.add_argument("--head_num",type=int,default=1, help="don't need to set its value")
parser.add_argument("--beta",type=float,default=0.05,help='it controls the local Attention')
parser.add_argument("--tau",type=float,default=0.6,help="tau for CL")
parser.add_argument('--dropout', type=float, default=0.6,help='Dropout rate.')
parser.add_argument('--smooth_num', type=int, default=3, help='initinal represention smooth degree') # 1,2,3,4
parser.add_argument('--C', type=int, default=1,help='global state num')  # 1,2,3
parser.add_argument('--maskingRate', type=float, default=0.3,help='dropout for CL')
parser.add_argument('--local', type=int, default=1,help='it controls the local attention')


args = parser.parse_args()
set_seed(args.seed)
args.cuda =args.cuda and torch.cuda.is_available()


data =   load_data(args.data)
nfeat = data.num_features
nclass = int(data.y.max())+1

all_test_acc = []
start_time = time.time()
for i in range(5):

    test_acc_list=[]
    net = getattr(models, args.model)(args, nfeat, nclass)
    net = net.cuda() if args.cuda==True else net.cpu()

    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
    criterion = torch.nn.CrossEntropyLoss()


    best_val=0
    best_test=0
    best_MAD = 0
    for epoch in range(args.epochs):
        train_loss, train_acc ,_ ,_= train(net, args.model,optimizer, criterion, data)
        test_acc,val_acc,_,feature_k = val_and_test(net, args.model,data)

        if best_val < val_acc:
            best_val = val_acc
            best_test = test_acc

        print("epoch:{} train_acc:{},train_loss:{},val_acc:{},test_acc:{},best_acc:{}".format(epoch,round(train_acc.tolist(),3),round(train_loss.tolist(),3),round(val_acc.tolist(),3),round(test_acc.tolist(),3),round(best_test.tolist(),3)))
    print("\ncurrent best test acc",best_test)
    all_test_acc.append(best_test)

end_time = time.time()
print("Time:",end_time-start_time)
print("\nall best test acc",all_test_acc)
print("\nMean test acc:{}+{}".format(torch.stack(all_test_acc,0).mean(0),torch.stack(all_test_acc,dim=0).std() ))
print("\nall best test acc",torch.stack(all_test_acc,dim=0).max())

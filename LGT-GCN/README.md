# Training GNN with ContraNorm

Official pytorch code for paper [**Layer-Wise Greedy Training GCN against Over-Smoothing**]

## Introduction
Our code was experimented with under torch 2.0.0 mitigation, you can also run the code in other suitable environments.

## Examples
Here are some of the commands we provide that you can use to run this code on pycharm.

```Cora
python main.py --data cora --model LGT_GCN --hid 128 --lr 0.05 --epochs 200 --wightDecay 0.0005 --nlayer 16 --seed 30 --alpha 1 --beta 0.05 --tau 0.5 --dropout 0.4 --head_num 1 --smooth_num 3 --C 2 --maskingRate 0.2 --local 1
```

```Citeseer
python main.py --data citeseer --model LGT_GCN --hid 128 --lr 0.05 --epochs 200 --wightDecay 0.0005 --nlayer 8 --seed 30 --alpha 1 --beta 0.05 --tau 0.6 --dropout 0.6 --head_num 1 --smooth_num 3 --C 1 --maskingRate 0.2 --local 1
```

```Pubmed
python main.py --data pubmed --model LGT_GCN --hid 128 --lr 0.05 --epochs 200 --wightDecay 0.0005 --nlayer 8 --seed 30 --alpha 1 --beta 0.05 --tau 0.8 --dropout 0.5 --head_num 1 --smooth_num 3 --C 2 --maskingRate 0.3 --local 1
```


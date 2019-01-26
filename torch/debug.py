import argparse
import numpy as np
import torch.cuda as torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import MKR
from train import get_data_for_kge, get_data_for_rs, train

np.random.seed(123)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--dim', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=2, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=1e-3, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
parser.add_argument('--cuda', action="store_true", default=False, help='set this to use cuda')
args = parser.parse_args()
data = load_data(args)

n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
# train_data: [user item score]
train_data, test_data = data[4], data[5]
# kg: [head relation tail]
kg = data[6]

mkr = MKR(args, n_user, n_item, n_entity, n_relation)
loss_func = nn.BCELoss()
optimizer_kge = optim.Adam(mkr.parameters(), lr=args.lr_kge)

feed, tail_indices = get_data_for_kge(kg, 0, args.batch_size)
feed, tail_indices = torch.Tensor(feed).long(), torch.Tensor(tail_indices).long()

print("feed", feed)
print("labels", tail_indices)

for _ in range(10):
    tail_pred = mkr("kge", feed)

    tail_embeddings = mkr.entity_emb_matrix(tail_indices)
    for i in range(mkr.L):
        tail_embeddings = mkr.tail_mlps[i](tail_embeddings)
    # 定义均方根误差
    rmse = (((tail_embeddings - tail_pred) ** 2).sum(1) / args.dim).sqrt().sum()
    print("rmse: {}".format(rmse.item()))

    optimizer_kge.zero_grad()
    rmse.backward()
    optimizer_kge.step()

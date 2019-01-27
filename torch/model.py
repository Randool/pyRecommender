import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from layers import CrossCompressUnit, Dense


class MKR(nn.Module):
    def __init__(self, args, n_users, n_items, n_entities, n_relations):
        super(MKR, self).__init__()
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations
        self.L = args.L
        self.H = args.H
        self.dim = args.dim

        # 定义embedding矩阵
        self.user_emb_matrix = nn.Embedding(n_users, args.dim)
        self.item_emb_matrix = nn.Embedding(n_items, args.dim)
        self.entity_emb_matrix = nn.Embedding(n_entities, args.dim)
        self.relation_emb_matrix = nn.Embedding(n_relations, args.dim)

        # 定义网络
        self.user_mlps, self.tail_mlps, self.cc_units = [], [], []
        self.kge_mlps = []
        for _ in range(args.L):
            self.user_mlps.append(Dense(args.dim, args.dim))
            self.tail_mlps.append(Dense(args.dim, args.dim))
            self.cc_units.append(CrossCompressUnit(args.dim))

        for _ in range(args.H):
            self.kge_mlps.append(Dense(args.dim * 2, args.dim * 2))

        self.kge_pred_mlp = Dense(args.dim * 2, args.dim)
        self.sigmoid = nn.Sigmoid()
    
    def cuda(self, device=None):
        for i in range(self.L):
            self.user_mlps[i].cuda()
            self.tail_mlps[i].cuda()
            self.cc_units[i].cuda()
        return self._apply(lambda t: t.cuda(device))

    def forward(self, mode: str, inputs: list):
        mode = mode.lower()

        if mode == "rs":
            user_indices, item_indices, head_indices = inputs
            # rs下层网络
            self.user_embeddings = self.user_emb_matrix(user_indices)
            for i in range(self.L):
                self.user_embeddings = self.user_mlps[i](self.user_embeddings)
        elif mode == "kge":
            item_indices, head_indices, relation_indices = inputs
            # kge下层网络
            relation_embeddings = self.relation_emb_matrix(relation_indices)

        # 共有下层网络
        self.item_embeddings = self.item_emb_matrix(item_indices)
        self.head_embeddings = self.entity_emb_matrix(head_indices)
        # 权值交换单元
        for i in range(self.L):
            self.item_embeddings, self.head_embeddings = self.cc_units[i](
                [self.item_embeddings, self.head_embeddings]
            )

        if mode == "rs":
            # rs上层网络
            # [batch_size]
            score = (self.user_embeddings * self.item_embeddings).sum(1)
            score = self.sigmoid(score)
            return score
        elif mode == "kge":
            # kge上层网络
            # [batch_size, dim * 2]
            head_relation_concat = torch.cat([self.head_embeddings, relation_embeddings], 1)
            for i in range(self.H - 1):
                head_relation_concat = self.kge_mlps[i](head_relation_concat)
            tail_pred = self.kge_pred_mlp(head_relation_concat)
            tail_pred = self.sigmoid(tail_pred)
            return tail_pred

    def evaluate(self, inputs, y_true):
        """
        测试RS的性能。
        inputs: user_indices, item_indices, head_indices
        return: AUC, accuracy
        """
        with torch.set_grad_enabled(False):
            y_scores = self.forward("rs", inputs)
            AUC = roc_auc_score(y_true.cpu(), y_scores.cpu())
            predicts = y_scores > 0.5
            accuracy = (predicts == y_true).float().mean()
        
        return AUC, accuracy

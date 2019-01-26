from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import MKR


def get_user_record(data, is_train: bool):
    """ 返回每个用户对应的一个喜好集合 """
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def get_data_for_rs(data, start, end):
    user_indices = data[start:end, 0]
    item_indices = data[start:end, 1]
    labels = data[start:end, 2]
    head_indices = data[start:end, 1]
    return (user_indices, item_indices, head_indices), labels


def get_data_for_kge(kg, start, end):
    item_indices = kg[start:end, 0]
    head_indices = kg[start:end, 0]
    relation_indices = kg[start:end, 1]
    tail_indices = kg[start:end, 2]
    return (item_indices, head_indices, relation_indices), tail_indices


def train(args, data):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    # train_data: [user item score]
    train_data, test_data = data[4], data[5]
    # kg: [head relation tail]
    kg = data[6]

    mkr = MKR(args, n_user, n_item, n_entity, n_relation)
    if args.cuda:
        mkr.cuda()

    loss_func = nn.BCELoss()
    optimizer_rs = optim.Adam(mkr.parameters(), lr=args.lr_rs)
    optimizer_kge = optim.Adam(mkr.parameters(), lr=args.lr_kge)

    # store best state
    best_test_acc = 0.0
    best_state_dict = None

    for epoch in range(args.n_epochs):
        # RS training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            feed, labels = get_data_for_rs(train_data, start, start + args.batch_size)
            feed, labels = torch.Tensor(feed).long(), torch.Tensor(labels).float()
            if args.cuda:
                feed, labels = feed.cuda(), labels.cuda()
            scores_normalized = mkr("rs", feed)

            # build loss for RS
            base_loss_rs = loss_func(scores_normalized, labels)
            l2_loss_rs = (mkr.user_embeddings ** 2).sum() / 2 + (mkr.item_embeddings ** 2).sum() / 2
            loss_rs = base_loss_rs + l2_loss_rs * args.l2_weight
            
            optimizer_rs.zero_grad()
            loss_rs.backward()
            optimizer_rs.step()

            start += args.batch_size

        if epoch % args.kge_interval == 0:        
            # KGE training
            np.random.shuffle(kg)
            start = 0
            while start < kg.shape[0]:
                feed, tail_indices = get_data_for_kge(kg, start, start + args.batch_size)
                feed, tail_indices = torch.Tensor(feed).long(), torch.Tensor(tail_indices).long()
                if args.cuda:
                    feed, tail_indices = feed.cuda(), tail_indices.cuda()
                tail_pred = mkr("kge", feed)
                
                # build loss for KGE
                tail_embeddings = mkr.entity_emb_matrix(tail_indices)
                for i in range(mkr.L):
                    tail_embeddings = mkr.tail_mlps[i](tail_embeddings)
                #scores_kge = mkr.sigmoid((tail_embeddings * tail_pred).sum(1))
                #l2_loss_kge = (mkr.head_embeddings ** 2).sum() / 2 + (tail_embeddings ** 2).sum() / 2
                #loss_kge = -scores_kge + l2_loss_kge * args.l2_weight
                rmse = (((tail_embeddings - tail_pred) ** 2).sum(1) / args.dim).sqrt().sum()

                optimizer_kge.zero_grad()
                rmse.backward()
                optimizer_kge.step()

                start += args.batch_size
    
        # Evaluating——train data
        inputs, y_true = get_data_for_rs(train_data, 0, train_data.shape[0])
        inputs, y_true = torch.Tensor(inputs).long(),torch.Tensor(y_true).byte()
        if args.cuda:
            inputs, y_true = inputs.cuda(), y_true.cuda()
        train_auc, train_acc = mkr.evaluate(inputs, y_true)

        # Evaluating——test data
        inputs, y_true = get_data_for_rs(test_data, 0, test_data.shape[0])
        inputs, y_true = torch.Tensor(inputs).long(), torch.Tensor(y_true).byte()
        if args.cuda:
            inputs, y_true = inputs.cuda(), y_true.cuda()
        test_auc, test_acc = mkr.evaluate(inputs, y_true)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state_dict = deepcopy(mkr.state_dict())
        
        print(
            "epoch {:3d} | train auc: {:.4f} acc: {:.4f} | test auc: {:.4f} acc:{:.4f}".format(
                epoch, train_auc, train_acc, test_auc, test_acc
            )
        )
        
    # Save model
    wts_name = "../model/MKR_{:.4f}.pth".format(best_test_acc)
    torch.save(best_state_dict, wts_name)
    print("Saved model to {}".format(wts_name))

    return mkr

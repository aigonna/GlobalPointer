#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 12:37
# @Author  : aigonna  
# @File    : losses.py
import torch
import numpy as np

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    代码第1, 2, 3行解释
    1.将真实标签 y_true 从0/1映射到-1/1，即将正类设为-1，负类设为1。
    并将得到的结果与预测值相乘。这一步处理的目的是为了保证预测值 y_pred
    落在 [0, 1] 的范围内。
    2.将正类位置的预测值设为负无穷。
    在这里采用了一个技巧，即将正类的预测值减去一个很大的数，这里是1e12。
    这样在经过softmax函数计算后，正类的概率会趋近于0，达到屏蔽正类的目的。
    3.将负类位置的预测值设为负无穷
    同样采用上述技巧，将负类的预测值减去一个很大的数，使得在经过softmax
    函数计算后，负类的概率会趋近于0，达到屏蔽负类的目的。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12 #mask pos
    y_pred_pos = y_pred - (1-y_true) * 1e12 #mask neg
    #构建y_pred[..., :1]一样形状的全0tensor来替换对应位置的预测值
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def pesudo_loss_fun(y_true, y_pred, pseudos):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    pseudos: [1, 1, 1, 0, 0, 0] 1是伪标签， 0是非伪标签
    """

    pseudos = torch.tensor(np.array(pseudos))
    pseudo = list(np.where(pseudos == 1)[0])
    normal = list(np.where(pseudos == 0)[0])
    loss = 0.0
    if len(pseudo) != 0:
        y_true1, y_pred1 = y_true[pseudo], y_pred[pseudo]
        batch_size, ent_type_size = y_true1.shape[:2]
        y_true1 = y_true1.reshape(batch_size * ent_type_size, -1)
        y_pred1 = y_pred1.reshape(batch_size * ent_type_size, -1)
        loss1 = multilabel_categorical_crossentropy(y_true1, y_pred1)
        loss += loss1 * 0.5

    if len(normal) != 0:
        y_true2, y_pred2 = y_true[normal], y_pred[normal]
        batch_size, ent_type_size = y_true2.shape[:2]
        y_true2 = y_true2.reshape(batch_size * ent_type_size, -1)
        y_pred2 = y_pred2.reshape(batch_size * ent_type_size, -1)
        loss2 = multilabel_categorical_crossentropy(y_true2, y_pred2)
        loss += loss2

    return loss


def loss_fn(y_true, y_pred):
    """
    calculate multilable loss
    :param y_true: (batch_size, ent_type_size, seq_len, seq_len)
    :param y_pred: (batch_size, ent_type_size, seq_len, seq_len)
    :return: multilabel categorical ce
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size*ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size*ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss
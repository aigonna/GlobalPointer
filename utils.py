#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 12:31
# @Author  : aigonna  
# @File    : utils.py
import os
import math
import time
import torch
import random
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup,\
                    get_cosine_schedule_with_warmup


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s %(message)s"))
    handler2 = FileHandler(filename=f"{filename}")
    handler2.setFormatter(Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def build_optimizer_and_scheduler(args, model, total_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    if args.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps
                                                    )
    else:
        scheduler = get_cosine_schedule_with_warmup(
                                                    optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps,
                                                    num_cycles=0.5
        )

    return optimizer, scheduler

#9类实体，具体参考https://tianchi.aliyun.com/dataset/144495
ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4,
              "ite": 5, "dep": 6, "dru": 7, "equ": 8}
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k

# print(len(ent2id))
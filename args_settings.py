#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 12:06
# @Author  : aigonna  
# @File    : args_settings.py
import argparse


def args_settings():
    parser = argparse.ArgumentParser(description='NER')
    parser.add_argument('--data_path', type=str, default='./data/CMeEE')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--save_args', type=bool, default=True)
    # model name: hfl/chinese-macbert-base    hfl/chinese-macbert-large
    # hfl/chinese-bert-wwm-ext    sijunhe/nezha-cn-base WENGSYX/CirBERTa-Chinese-Base
    # trueto/medbert-kd-chinese
    parser.add_argument('--model_name', type=str, default=r'hfl/chinese-macbert-large')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--epochs', default=15, type=int)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1000)
    parser.add_argument('--apex', type=bool, default=True)
    parser.add_argument('--batch_scheduler', type=bool, default=True)
    parser.add_argument('--logging_steps', default=100, type=int)

    parser.add_argument('--eval_steps', default=5000, type=int)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--scheduler_type', type=str, default='cosine')

    #gp epg gp_spatial_drop
    parser.add_argument('--gp_type', type=str, default="epg_spatial_drop")
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--fgm', type=bool, default=True)
    parser.add_argument('--awp', type=bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--adv_lr', type=float, default=1.0)
    parser.add_argument('--adv_eps', type=float, default=0.2)

    args = parser.parse_args([])
    return args


def args_pred():
    parser = argparse.ArgumentParser(description='NER_pred')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--model_name', type=str, default=r'hfl/chinese-macbert-base')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--spatialdrop', type=bool, default=True)

    args = parser.parse_args()
    return args

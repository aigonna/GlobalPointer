#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 12:25
# @Author  : aigonna  
# @File    : train.py
import os
import json
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import loss_fn
from args_settings import args_settings
from data_utils import load_data, EntDataset
from globalpointer import MetricsCalculator, GlobalPointer, GlobalPointerSdrop,\
                    EffiGlobalPointer, EffiGlobalPointerSdrop
from adversarial import EMA, FGM, AWP
from utils import seed_everything, get_logger, build_optimizer_and_scheduler,\
            ent2id, AverageMeter, timeSince
from transformers import AutoTokenizer, AutoModel


def evaluate(model, valid_loader, metrics, device):
    model.eval()

    eval_metrics = {}

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch in tqdm(valid_loader, desc='Evalation'):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, segment_ids)

        f1, p, r = metrics.get_evaluate_fpr(logits, labels)

        total_f1 += f1
        total_precision += p
        total_recall += r

    avg_f1 = total_f1 / len(valid_loader)
    avg_precision = total_precision / len(valid_loader)
    avg_recall = total_recall / len(valid_loader)

    eval_metrics['f1'] = avg_f1
    eval_metrics['precision'] = avg_precision
    eval_metrics['recall'] = avg_recall

    return eval_metrics


def train_loop(args, logger, device):
    # Dataset
    train_path = os.path.join(args.data, "CMeEE/CMeEE_train.json")
    valid_path = os.path.join(args.data, "CMeEE/CMeEE_dev.json")

    train_set = load_data(train_path, ent2id)
    valid_set = load_data(valid_path, ent2id)
    print(f"Train dataset length: {len(train_set)}")
    print(f"Valid dataset length: {len(valid_set)}")
    print("Train dataset sample 1:", train_set[1])
    # 一个示例:
    # ['支气管软化者应注意体位引流，可应用色甘酸、溴化异丙托品，但应避免使用β受体激动剂。',
    # (0, 4, 1), (9, 12, 4), (17, 19, 7), (21, 26, 7), (34, 39, 7)]
    # 元组表示start_idx, end_idx, 实体类别
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)

    if args.gp_type == "gp":
        model = GlobalPointer(encoder, len(ent2id), 64).to(device)
    elif args.gp_type == "gps":
        model = GlobalPointerSdrop(encoder, len(ent2id), 64).to(device)
    elif args.gp_type == "egp":
        model = EffiGlobalPointer(encoder, len(ent2id), 64).to(device)
    else:
        model = EffiGlobalPointerSdrop(encoder, len(ent2id), 64).to(device)

    if args.ema:
        model.ema = EMA(model, args.ema_decay, device)
        model.ema.register()

    train_set = EntDataset(train_set, tokenizer, args.max_len, len(ent2id))
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              collate_fn=train_set.collate,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    valid_set = EntDataset(valid_set, tokenizer, args.max_len, len(ent2id))
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              collate_fn=valid_set.collate,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True)

    #     print(len(train_set) * args.epochs)

    total_steps = int(len(train_set) / args.batch_size * args.epochs /
                      args.gradient_accumulation_steps)

    optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_steps)

    metrics = MetricsCalculator()
    best_f1, best_f1_ema = 0.0, 0.0

    for epoch in range(1, args.epochs):
        losses = AverageMeter()
        f1_scores = AverageMeter()
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
        start = time.time()
        if args.fgm:
            fgm = FGM(model)

        if args.awp:
            awp = AWP(model,
                      optimizer,
                      adv_lr=args.adv_lr,
                      adv_eps=args.adv_eps,
                      scaler=scaler)
        for idx, batch in enumerate(train_loader):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)

            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=args.apex):
                logits = model(input_ids, attention_mask, segment_ids)
                loss = loss_fn(logits, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if args.fgm:
                fgm.attack()
                with torch.cuda.amp.autocast(enabled=args.apex):
                    logits_adv = model(input_ids, attention_mask, segment_ids)
                    loss_adv = loss_fn(logits_adv, labels)
                    loss_adv.backward()
                fgm.restore()

                # Adversarial Weight Perturbation (AWP)
            if args.awp:
                loss_awp = awp.attack_backward(input_ids, labels,
                                               attention_mask, segment_ids, epoch)
                loss_awp.backward()
                awp._restore()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if args.batch_scheduler:
                    scheduler.step()

                if args.ema:
                    model.ema.update()

            sample_f1 = metrics.get_sample_f1(logits, labels)

            losses.update(loss.item(), batch_size)
            f1_scores.update(sample_f1.item(), batch_size)

            if (idx + 1) % args.logging_steps == 0 or idx == (len(train_loader) - 1):
                print(
                    f"Epoch: {epoch:>2d}| {idx + 1:>4d}/{len(train_loader):>4d}, "
                    f"train_loss: {losses.avg:5.2f}, train_f1: {f1_scores.avg:.6f}, "
                    f"lr: {scheduler.get_lr()[0]:.4e}, grad_norm: {grad_norm.item():5.2f},"
                    f" Elapsed: {timeSince(start, float(idx + 1) / len(train_loader))}"
                )

        valid_scores = evaluate(model, valid_loader, metrics, device)

        logger.info(f"Epoch: {epoch:>2d}, valid f1: {valid_scores['f1']:.4f}, "
                    f"valid_precision: {valid_scores['precision']:.4f}, "
                    f"valid_recall: {valid_scores['recall']:.4f}")

        if valid_scores['f1'] > max(best_f1, best_f1_ema):
            best_f1 = valid_scores['f1']
            print(f"Saved best model with F1 score:{best_f1:.4f}")
            torch.save(model.state_dict(), os.path.join(args.output, 'best.pt'))

        if args.ema:
            model.ema.apply_shadow()
            # scoring
            valid_scores = evaluate(model, valid_loader, metrics, device)

            if valid_scores['f1'] > max(best_f1, best_f1_ema):
                best_f1_ema = valid_scores['f1']
                print(f"Saved best model with F1 score:{best_f1_ema:.4f}")
                torch.save(model.state_dict(), os.path.join(args.output, 'best.pt'))
            model.ema.restore()

    logger.info(f"Training best score is {max(best_f1, best_f1_ema)}.")
    logger.info(f"Training {args.epochs} epoch done!")


def main():
    args = args_settings()
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))

    seed_everything(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.save_args:
        with open(os.path.join(args.output, 'args.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)

    logger = get_logger(args.output + '/train.log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loop(args, logger, device)


if __name__ == '__main__':
    main()

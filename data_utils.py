#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 12:05
# @Author  : aigonna  
# @File    : data_utils.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def load_data(path, ent2id):
    datalist = []
    for d in json.load(open(path, encoding='utf-8')):
        datalist.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                datalist[-1].append((start, end, ent2id[label]))
    return datalist

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, len_ent, istrain=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len_ent = len_ent
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True,
                                                     max_length=self.max_len, truncation=True)["offset_mapping"]
            start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            #将raw_text的下标 与 token的start和end下标对应
            encoder_txt = self.tokenizer.encode_plus(text, max_length=self.max_len, truncation=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask
        else:
            #TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)

            labels = np.zeros((self.len_ent, self.max_len, self.max_len))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels

    def __getitem__(self, index):
        item = self.data[index]
        return item

if __name__ == '__main__':
    #https://tianchi.aliyun.com/dataset/144495
    #https://github.com/xhw205/GlobalPointer_torch/tree/main
    from transformers import AutoTokenizer
    max_len = 256
    ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4,
              "ite": 5, "dep": 6, "dru": 7, "equ": 8}
    id2ent = {}
    for k, v in ent2id.items():
        id2ent[v] = k


    train_cme_path = './data/CMeEE/CMeEE_train.json'  # CMeEE 训练集
    eval_cme_path = './data/CMeEE/CMeEE_dev.json'  # CMeEE 测试集

    train_set = load_data(train_cme_path, ent2id)
    valid_set = load_data(eval_cme_path, ent2id)
    print(train_set)
    #一个示例:
    #['支气管软化者应注意体位引流，可应用色甘酸、溴化异丙托品，但应避免使用β受体激动剂。',
    # (0, 4, 1), (9, 12, 4), (17, 19, 7), (21, 26, 7), (34, 39, 7)]

    model_name = "hfl/chinese-roberta-wwm-ext"
    # model_name = r"D:\Anaconda\amodels\nezha\nezha-cn-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_set = EntDataset(train_set, tokenizer=tokenizer,
                           max_len=max_len, len_ent=len(ent2id))
    train_loader = DataLoader(train_set, batch_size=2,
                              collate_fn=train_set.collate,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

    valid_set = EntDataset(valid_set, tokenizer=tokenizer,
                           max_len=max_len, len_ent=len(ent2id))
    valid_loader = DataLoader(valid_set, batch_size=2,
                              collate_fn=train_set.collate,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    print(next(iter(valid_loader)))


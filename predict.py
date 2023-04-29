#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 17:56
# @Author  : aigonna  
# @File    : predict.py
import os
import torch
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer
from globalpointer import GlobalPointer, GlobalPointerSdrop
from args_settings import args_pred
from utils import ent2id, id2ent, seed_everything
from tqdm import tqdm


def predict(text, tokenizer, model, max_len=256):
    token2char_span_mapping = tokenizer(text,
                                        return_offsets_mapping=True,
                                        max_length=max_len)["offset_mapping"]
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda()
    scores = model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx": new_span[start][0],
                         "end_idx": new_span[end][-1],
                         "type": id2ent[l]})

    return {"text": text, "entities": entities}


def main():
    args = args_pred()
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    if args.spatialdrop:
        model = GlobalPointerSdrop(encoder, len(ent2id), 64).to(device)
    else:
        model = GlobalPointer(encoder, len(ent2id), 64).to(device)
    path = os.path.join(args.output, 'best.pt')

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()

    all_ = []
    test_path = os.path.join(args.data, './CMeEE/CMeEE_test.json')
    for d in tqdm(json.load(open(test_path))):
        all_.append(predict(d["text"], tokenizer, model, args.max_len))
    json.dump(all_,
        open('./outputs/CMeEE_test.json', 'w'),
        indent=4,
        ensure_ascii=False
    )



if __name__ == '__main__':
    main()
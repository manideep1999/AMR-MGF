import os
import sys

sys.path.append(r'./LAL-Parser/src_joint')
import re
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['amr_tokens'])
                tok = list(d['amr_tokens'])  # word token
                length = len(tok)  # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])  # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']  # label
                pos = list(d['pos'])  # pos_tag
                head = list(d['head'])  # head
                deprel = list(d['deprel'])  # deprel
                amr_edges = list(d['amr_edges'])
                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]  # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head, \
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list, 'amr_edges': amr_edges}
                all_data.append(sample)

    return all_data

def get_embedding(vocab, opt):
    graph_emb=0
    if 'laptop' in opt.dataset:
        graph_file = 'APARN/embeddings/entity_embeddings_analogy_400.txt'
        if opt.is_bert==0:
            graph_pkl = 'APARN/embeddings/%s_graph_analogy.pkl' % opt.dataset
        else:
            graph_pkl = 'APARN/embeddings/%s_graph_analogy_bert.pkl' % opt.dataset
        # graph_pkl = 'APARN/embeddings/%s_graph_analogy_roberta.pkl' % ds_name
    elif 'restaurant' in opt.dataset:
        graph_file = 'APARN/embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'APARN/embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'APARN/embeddings/%s_graph_dismult_bert.pkl' % opt.dataset
        graph_pkl = 'APARN/embeddings/%s_graph_dismult_roberta.pkl' % opt.dataset  # Updated line
    elif 'twitter' in opt.dataset:
        graph_file = 'APARN/embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'APARN/embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'APARN/embeddings/%s_graph_dismult_bert.pkl' % opt.dataset

    if not os.path.exists(graph_pkl):
        graph_embeddings = np.zeros((len(vocab)+1, opt.dim_k), dtype='float32')
        with open(graph_file, encoding='utf-8') as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                graph_emb += 1
                if w in vocab:
                    try:
                        graph_embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        pass
        pickle.dump(graph_embeddings, open(graph_pkl, 'wb'))
    else:
        graph_embeddings = pickle.load(open(graph_pkl, 'rb'))

    return graph_embeddings

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        stoi = torch.load(opt.amr_edge_stoi)
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            deprel = obj['deprel']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            amr_edge_adj = np.ones((len(text_list), len(text_list)), dtype=int) * stoi.get('none')
            try:
                for edge in obj['amr_edges']:
                    if edge[1].startswith(':op') and not stoi.get('Ġ' + edge[1]):
                        edge[1] = ':op5'
                    if edge[1].startswith(':snt') and not stoi.get('Ġ' + edge[1]):
                        edge[1] = ':snt5'
                    if edge[1] == ':prep-on-behalf-of':
                        edge[1] = 'behalf'
                    if edge[1] == ':prep-in-addition-to':
                        edge[1] = 'addition'
                    if edge[1] == ':prep-along-with':
                        edge[1] = 'along'
                    if edge[1].startswith(':prep-'):
                        amr_edge_adj[edge[0]][edge[2]] = stoi.get('Ġ' + edge[1][6:]) \
                            if amr_edge_adj[edge[0]][edge[2]] == stoi.get('none') else amr_edge_adj[edge[0]][
                            edge[2]]
                    elif edge[1].endswith('-of'):
                        amr_edge_adj[edge[2]][edge[0]] = stoi.get('Ġ' + edge[1][:-3]) \
                            if amr_edge_adj[edge[2]][edge[0]] == stoi.get('none') else amr_edge_adj[edge[2]][
                            edge[0]]
                    else:
                        amr_edge_adj[edge[0]][edge[2]] = stoi.get('Ġ' + edge[1]) \
                            if amr_edge_adj[edge[0]][edge[2]] == stoi.get('none') else amr_edge_adj[edge[0]][
                            edge[2]]
            except Exception:
                print(obj['amr_edges'])
                print(edge)
                raise Exception("Error!")
            amr_edge_adj -= np.diag(np.diag(amr_edge_adj))
            # if not opt.direct:
            #     amr_edge_adj = amr_edge_adj + amr_edge_adj.T
            amr_edge_adj = amr_edge_adj + np.eye(amr_edge_adj.shape[0]) * stoi.get('self')

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map 
            truncate_tok_len = len(bert_tokens)
            edge_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    edge_adj[i][j] = amr_edge_adj[tok2ori_map[i]][tok2ori_map[j]]

            concat_bert_indices = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(bert_tokens) + \
                                  [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                                    [tokenizer.sep_token_id]
            context_asp_len = len(concat_bert_indices)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            concat_segments_indices = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            concat_bert_indices += paddings
            concat_bert_indices = np.asarray(concat_bert_indices, dtype='int64')
            concat_segments_indices = np.asarray(concat_segments_indices, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')

            # pad edge adj
            edg_adj_matrix = np.ones((tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('int64')
            edge_pad_adj = np.ones((context_asp_len, context_asp_len)).astype('int64')
            if not opt.edge == "same":
                edge_pad_adj[1:context_len + 1, 1:context_len + 1] = edge_adj
            edg_adj_matrix[:context_asp_len, :context_asp_len] = edge_pad_adj

            text_bert_indices = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(bert_tokens) + [tokenizer.sep_token_id] 
            aspect_bert_indices = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [tokenizer.sep_token_id]
            text_bert_len = len(text_bert_indices)
            asp_ber_len = len(aspect_bert_indices)

            text_bert_indices += [0] * (tokenizer.max_seq_len - len(text_bert_indices))
            text_bert_indices = np.asarray(text_bert_indices, dtype='int64')
            aspect_bert_indices += [0] * (tokenizer.max_seq_len - len(aspect_bert_indices))
            aspect_bert_indices = np.asarray(aspect_bert_indices, dtype='int64')                        
            edg_adj_matrix = np.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('int64')
            edg_adj_matrix[1:context_len + 1, 1:context_len + 1] = edge_adj
            text_attention_mask = [1] * text_bert_len + [0] * (tokenizer.max_seq_len - text_bert_len)
            text_attention_mask = np.asarray(text_attention_mask, dtype='int64')
            asp_attention_mask = [1] * asp_ber_len + [0] * (tokenizer.max_seq_len - asp_ber_len)  
            asp_attention_mask = np.asarray(asp_attention_mask, dtype='int64')
            aspect_combined_terms = ' '.join([' '.join(t) if isinstance(t, list) else t for t in term])

            # Pad tok2ori_map to this maximum length
            tok2ori_map_padded = tok2ori_map + [0] * (tokenizer.max_seq_len - len(tok2ori_map))
            tok2ori_map_padded = np.asarray(tok2ori_map_padded, dtype='int64')
            
            # 1、tok_length 
            tok_length = tok2ori_map_padded[-1] + 1
            # 2、bert_length
            bert_length = len(tok2ori_map_padded) 

            data = {
                'text': text,
                'aspect': aspect_combined_terms,
                'tok_length': tok_length,
                'bert_length': bert_length,
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'attention_mask': context_asp_attention_mask,
                'text_attention_mask': text_attention_mask,
                'asp_attention_mask': asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'edge_adj': edg_adj_matrix,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'tok2ori_map': tok2ori_map_padded,
                'polarity': polarity,
            }
            self.data.append(data)
        if opt.part <= 1 and "train" in fname:
            self.data = random.sample(self.data, int(len(self.data) * opt.part))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     # Debug: Print shape information for tensor fields
    #     for key, value in item.items():
    #         if isinstance(value, np.ndarray):
    #             print(f"Sample {idx}, key {key}: shape {value.shape}")
    #         elif isinstance(value, list) and len(value) > 0:
    #             print(f"Sample {idx}, key {key}: list length {len(value)}")
    #     return item
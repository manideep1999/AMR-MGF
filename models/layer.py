import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import pickle
import copy
import numpy as np
from torch_geometric.nn import GCNConv, GCN2Conv, TAGConv, ChebConv, GatedGraphConv, ResGatedGraphConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


# con, dep and seman update module
class GCNEncoder(nn.Module):
    def __init__(self, emb_dim, con_layers, dep_layers, sem_layers, amr_layers = 9, kg_dim=200, gcn_dropout=0.1):
        super().__init__()
        self.con_layers = con_layers
        self.dep_layers = dep_layers
        self.sem_layers = sem_layers
        self.amr_layers = amr_layers

        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.kg_dim = kg_dim
        
        # Add text to KG projection layer
        self.text_to_kg_proj = nn.Linear(emb_dim, kg_dim)
        
        # gcn layer
        self.W_con = nn.ModuleList()
        self.W_dep = nn.ModuleList()
        self.W_sem = nn.ModuleList()
        self.W_amr = nn.ModuleList()
        for layer in range(self.con_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_con.append(nn.Linear(input_dim, input_dim))
        for layer in range(self.dep_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_dep.append(nn.Linear(input_dim, input_dim))
        for layer in range(self.sem_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_sem.append(nn.Linear(input_dim, input_dim))
        for layer in range(self.amr_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_amr.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)        

    def forward(self, inputs, con_adj, dep_adj, seman_adj, amr_adj, tok_length, knowledge_embeddings=None, opt=None):
        # gcn layer
        con_input, dep_input, seman_input, amr_input = inputs, inputs, inputs, inputs
        
        # denom_con_dep_seman
        dep_denom = dep_adj.sum(2).unsqueeze(2) + 1
        seman_denom = seman_adj.sum(2).unsqueeze(2) + 1
        amr_denom = amr_adj.sum(2).unsqueeze(2) + 1

        for index in range(self.con_layers):
            con_adj_new = con_adj[index].bool().float()
            con_denom = con_adj_new.sum(2).unsqueeze(2) + 1

            # con
            con_Ax = con_adj_new.bmm(con_input)
            con_AxW = self.W_con[index](con_Ax)
            con_AxW = con_AxW + self.W_con[index](con_input)  # self loop
            con_AxW = con_AxW / con_denom
            con_gAxW = F.relu(con_AxW)
            con_input = self.gcn_drop(con_gAxW) if index < self.con_layers - 1 else con_gAxW

        for index in range(self.dep_layers):
            # dep
            dep_Ax = dep_adj.bmm(dep_input)
            dep_AxW = self.W_dep[index](dep_Ax)
            dep_AxW = dep_AxW + self.W_dep[index](dep_input)  # self loop
            dep_AxW = dep_AxW / dep_denom
            dep_gAxW = F.relu(dep_AxW)
            dep_input = self.gcn_drop(dep_gAxW) if index < self.dep_layers - 1 else dep_gAxW

        for index in range(self.sem_layers):
            # seman
            seman_Ax = seman_adj.bmm(seman_input)
            seman_AxW = self.W_sem[index](seman_Ax)
            seman_AxW = seman_AxW + self.W_sem[index](seman_input)  # self loop
            seman_AxW = seman_AxW / seman_denom
            seman_gAxW = F.relu(seman_AxW)
            seman_input = self.gcn_drop(seman_gAxW) if index < self.sem_layers - 1 else seman_gAxW

        for index in range(self.amr_layers):
            amr_Ax = amr_adj.bmm(amr_input)
            amr_AxW = self.W_amr[index](amr_Ax)
            amr_AxW = amr_AxW + self.W_amr[index](amr_input)  # self-loop
            amr_AxW = amr_AxW / amr_denom
            amr_gAxW = F.relu(amr_AxW)
            amr_input = self.gcn_drop(amr_gAxW) if index < self.amr_layers - 1 else amr_gAxW

        # choose the text branch you want to align to KG (dep_input is fine)
        kg_loss = torch.zeros((), device=inputs.device)
        if knowledge_embeddings is not None:
            # Optional: warm-up to avoid early domination
            kg_temp       = getattr(opt, 'kg_temperature', 0.07)
            kg_sym        = getattr(opt, 'kg_symmetrize', True)
            kg_detach     = getattr(opt, 'kg_detach_teacher', True)
            kg_ls         = getattr(opt, 'kg_label_smoothing', 0.0)
            kg_weight     = getattr(opt, 'lambda_kg', 1.0)
            warm_steps    = getattr(opt, 'kg_warmup_steps', 0)
            global_step   = getattr(opt, 'global_step', 0)

            kg_w = kg_weight
            if warm_steps and warm_steps > 0:
                # linear ramp from 0 -> kg_weight
                kg_w = kg_weight * min(1.0, float(global_step) / float(warm_steps))

            kg_loss_val = kg_alignment_loss_inbatch(
                text_feats=dep_input,                 # or your fused text reps
                kg_feats=knowledge_embeddings,
                tok_len=tok_length,
                text_to_kg_proj=self.text_to_kg_proj,
                temperature=kg_temp,
                symmetrize=kg_sym,
                detach_kg=kg_detach,
                label_smoothing=kg_ls,
            )
            kg_loss = kg_w * kg_loss_val
        # print(amr_input)
        multi_loss = process_adj_matrices(dep_adj, con_adj[0].bool().float(), seman_adj, amr_adj, con_input, dep_input, amr_input, tok_length, knowledge_embeddings=None, opt=opt)
        # print(multi_loss)
        if opt is not None:
            kg_loss *= getattr(opt, 'lambda_kg_scalar', 1.0)
        total_loss = multi_loss + kg_loss
        # print('multi_loss:', multi_loss.item(), 'kg_loss:', kg_loss.item(), 'total_loss:', total_loss.item())
        return con_input, dep_input, seman_input, amr_input, total_loss    
    
def process_adj_matrices(dep_adj, con_adj_new, seman_adj, amr_adj, con_input, dep_input, amr_input, tok_len, knowledge_embeddings=None, opt=None):
    batch_size, max_length, _ = dep_adj.size()
    
    multi_viewdep_loss = 0.0
    multi_viewamr_loss = 0.0

    dep_denom = dep_adj.sum(2).unsqueeze(2) + 1  # (B,T,1) if needed later
    amr_denom = amr_adj.sum(2).unsqueeze(2) + 1

    for b in range(batch_size):
        length = int(tok_len[b])

        dep_adj_batch  = dep_adj[b, :length, :length].to(dtype=torch.int)
        con_adj_batch  = con_adj_new[b, :length, :length].to(dtype=torch.int)
        sem_adj_batch  = seman_adj[b, :length, :length]
        amr_adj_batch  = amr_adj[b, :length, :length].to(dtype=torch.int)

        # importance
        node_importance_scores = sem_adj_batch.mean(dim=1) + sem_adj_batch.max(dim=1).values
        k = max(1, int((math.log10(max(2, length))) ** 2))  # guard
        top_k_node = torch.topk(node_importance_scores, k).indices.tolist()

        dep_non_zero_indices = torch.nonzero(dep_adj_batch)
        con_non_zero_indices = torch.nonzero(con_adj_batch)
        dep_edges_tuple = dep_non_zero_indices.t()
        con_edges_tuple = con_non_zero_indices.t()

        dep_1_start = dep_edges_tuple[0]
        dep_1_end   = dep_edges_tuple[1]
        dep_3_start, dep_3_end = get_2nd_order_pairs(con_edges_tuple, dep_edges_tuple)

        dep_loss_sum = 0.0
        amr_loss_sum = 0.0

        for i in top_k_node:
            # ---- DEP-CON contrast ----
            Anchor_Dep = dep_input[b, i]
            AD_dep_view_node_index = dep_1_end[dep_1_start == i]
            AD_con_view_node_index = dep_3_end[dep_3_start == i]

            AD_dep_view_node = dep_input[b, AD_dep_view_node_index] if AD_dep_view_node_index.numel() > 0 else Anchor_Dep.unsqueeze(0)
            # negatives are "everything else" (simple, stable mask)
            all_idx = torch.arange(length, device=dep_input.device)
            dep_neg_mask = torch.ones(length, dtype=torch.bool, device=dep_input.device)
            dep_neg_mask[AD_dep_view_node_index] = False
            dep_neg = dep_input[b, all_idx[dep_neg_mask]]

            AD_con_view_node_index = torch.cat((AD_con_view_node_index, torch.tensor([i], device=dep_input.device)))
            AD_con_view_node = con_input[b, AD_con_view_node_index] if AD_con_view_node_index.numel() > 0 else Anchor_Dep.unsqueeze(0)
            con_neg_mask = torch.ones(length, dtype=torch.bool, device=dep_input.device)
            con_neg_mask[AD_con_view_node_index] = False
            con_neg = dep_input[b, all_idx[con_neg_mask]]

            AD_P = torch.cat((AD_dep_view_node, AD_con_view_node), dim=0)
            AD_N = torch.cat((dep_neg, con_neg), dim=0)

            dep_loss_sum += multi_margin_contrastive_loss(Anchor_Dep, AD_P, AD_N)

            # ---- AMR contrast ----
            Anchor_AMR = amr_input[b, i]
            amr_pos_idx = (amr_adj_batch[i] > 0).nonzero(as_tuple=True)[0]
            amr_pos_idx = amr_pos_idx[amr_pos_idx != i]
            AMR_pos = amr_input[b, amr_pos_idx] if amr_pos_idx.numel() > 0 else Anchor_AMR.unsqueeze(0)

            amr_all = torch.arange(length, device=amr_input.device)
            amr_neg_idx = amr_all[~torch.isin(amr_all, amr_pos_idx)]
            AMR_neg = amr_input[b, amr_neg_idx] if amr_neg_idx.numel() > 0 else Anchor_AMR.unsqueeze(0)

            amr_loss_sum += multi_margin_contrastive_loss(Anchor_AMR, AMR_pos, AMR_neg)

        if opt is not None:
            dep_loss_sum *= getattr(opt, 'lambda_dep', 1.0)
            amr_loss_sum *= getattr(opt, 'lambda_amr', 1.0)

        multi_viewdep_loss += dep_loss_sum
        multi_viewamr_loss += amr_loss_sum

    # return DEP/AMR; KG will be computed in-batch elsewhere
    return multi_viewdep_loss + multi_viewamr_loss

def _flatten_valid_tokens(x, tok_len):
    """
    x: (B, T, D), tok_len: (B,)
    returns: (N, D) flattened valid tokens
    """
    B, T, D = x.size()
    mask = torch.arange(T, device=x.device).unsqueeze(0) < tok_len.unsqueeze(1)  # (B, T)
    return x[mask]  # (N, D)

def knowledge_text_alignment_loss(text_anchor, kg_anchor, kg_positives, kg_negatives, margin=0.3, temperature=0.5):
    """
    SIMPLIFIED and STABLE text-knowledge alignment loss with better gradient flow
    """
    # Simplified deterministic dimension matching
    if text_anchor.size(-1) != kg_anchor.size(-1):
        target_dim = kg_anchor.size(-1)
        source_dim = text_anchor.size(-1)
        
        if source_dim > target_dim:
            # Simple linear projection for downsampling (learnable but fixed)
            # Use mean pooling over chunks for better information preservation
            chunk_size = source_dim // target_dim
            text_anchor_proj = text_anchor.view(-1, chunk_size).mean(dim=1)[:target_dim]
        else:
            # Simple repetition for upsampling
            text_anchor_proj = text_anchor.repeat((target_dim + source_dim - 1) // source_dim)[:target_dim]
    else:
        text_anchor_proj = text_anchor
    
    # Normalize all features for cosine similarity
    text_norm = F.normalize(text_anchor_proj, p=2, dim=-1, eps=1e-8)
    kg_anchor_norm = F.normalize(kg_anchor, p=2, dim=-1, eps=1e-8)
    kg_pos_norm = F.normalize(kg_positives, p=2, dim=-1, eps=1e-8)
    kg_neg_norm = F.normalize(kg_negatives, p=2, dim=-1, eps=1e-8)
    
    # Compute similarities
    text_kg_sim = F.cosine_similarity(text_norm.unsqueeze(0), kg_anchor_norm.unsqueeze(0), dim=-1)
    pos_sims = F.cosine_similarity(text_norm.unsqueeze(0), kg_pos_norm, dim=-1)
    neg_sims = F.cosine_similarity(text_norm.unsqueeze(0), kg_neg_norm, dim=-1)
    
    # STABLE: Use InfoNCE-style contrastive loss with proper temperature
    # Compute logits for all positive and negative samples
    all_sims = torch.cat([pos_sims, neg_sims], dim=0) / temperature
    
    # Create labels: 1 for positives, 0 for negatives
    num_pos = pos_sims.size(0)
    num_neg = neg_sims.size(0)
    labels = torch.zeros(num_pos + num_neg, device=all_sims.device)
    labels[:num_pos] = 1.0
    
    # InfoNCE loss: encourage high similarity with positives, low with negatives
    pos_logits = all_sims[:num_pos]
    neg_logits = all_sims[num_pos:]
    
    # Compute InfoNCE loss more stably
    pos_exp = torch.exp(pos_logits).mean()
    neg_exp = torch.exp(neg_logits).mean()
    infonce_loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
    
    # Simple margin loss for text-kg anchor alignment
    margin_loss = torch.clamp(margin - text_kg_sim, min=0).mean()
    
    # Balanced combination with reduced weights for stability
    total_loss = 0.6 * infonce_loss + 0.4 * margin_loss
    
    # Add small regularization to prevent mode collapse
    diversity_loss = -torch.var(pos_sims) * 0.01  # Encourage diversity in positive similarities
    
    return total_loss + diversity_loss

def kg_alignment_loss_inbatch(
    text_feats,               # (B, T, D_text)  e.g., dep_input or your fused text reps
    kg_feats,                 # (B, T, D_kg)    external KG embeddings aligned to tokens
    tok_len,                  # (B,)
    text_to_kg_proj,          # nn.Linear(D_text -> D_kg)
    temperature=0.07,
    symmetrize=True,
    detach_kg=True,
    label_smoothing=0.0,
):
    """
    In-batch InfoNCE between projected text and KG.
    Positive = same position token pair; Negatives = all other tokens in batch.
    We detach KG branch to stabilize (teacher signal).
    """
    # 1) gather only valid tokens across batch
    t_flat = _flatten_valid_tokens(text_feats, tok_len)          # (N, D_text)
    k_flat = _flatten_valid_tokens(kg_feats, tok_len)            # (N, D_kg)
    if t_flat.numel() == 0:
        return torch.zeros((), device=text_feats.device, dtype=text_feats.dtype)

    # 2) project text into KG dim and L2-normalize both
    t_proj = text_to_kg_proj(t_flat)                             # (N, D_kg)
    t_proj = torch.nn.functional.normalize(t_proj, p=2, dim=-1)
    if detach_kg:
        k_flat = k_flat.detach()
    k_flat = torch.nn.functional.normalize(k_flat, p=2, dim=-1)

    # 3) similarity logits (N x N) with temperature
    logits = (t_proj @ k_flat.t()) / temperature                 # (N, N)

    # 4) diagonal are the positives
    targets = torch.arange(logits.size(0), device=logits.device)

    # optional label smoothing
    if label_smoothing and label_smoothing > 0.0:
        # manual LS: one-hot with smoothing
        eps = label_smoothing
        N = logits.size(0)
        with torch.no_grad():
            soft_targets = torch.full_like(logits, fill_value=eps / (N - 1))
            soft_targets.scatter_(1, targets.view(-1,1), 1.0 - eps)
        log_probs = torch.log_softmax(logits, dim=1)
        loss_t2k = -(soft_targets * log_probs).sum(dim=1).mean()
    else:
        loss_t2k = torch.nn.functional.cross_entropy(logits, targets)

    if symmetrize:
        # KG->Text direction (share same targets)
        logits_k2t = logits.t()
        if label_smoothing and label_smoothing > 0.0:
            log_probs_kt = torch.log_softmax(logits_k2t, dim=1)
            loss_k2t = -(soft_targets * log_probs_kt).sum(dim=1).mean()
        else:
            loss_k2t = torch.nn.functional.cross_entropy(logits_k2t, targets)
        return 0.5 * (loss_t2k + loss_k2t)
    else:
        return loss_t2k

def multi_margin_contrastive_loss(anchor, positives, negatives, margin=0.2):
    dist_pos = F.pairwise_distance(anchor.unsqueeze(0), positives).mean()
    dist_neg = F.pairwise_distance(anchor.unsqueeze(0), negatives).mean()
    loss = torch.relu(dist_pos - dist_neg + margin) / 10

    return loss

def get_2nd_order_pairs(edge_list1, edge_list2):
    list1, list2 = [], []
    edge_list1 = edge_list1.tolist()  # 转换为Python列表
    edge_list2 = edge_list2.tolist()

    for x0, y0 in zip(edge_list1[0], edge_list1[1]):
        edge_exist = False
        for x1, y1 in zip(edge_list2[0], edge_list2[1]):
            if x1 == x0 and y1 == y0:
                edge_exist = True
                break
        if not edge_exist:
            list1.append(x0)
            list2.append(y0)
    list1, list2 = torch.tensor(list1).cuda(), torch.tensor(list2).cuda()
    list1, list2 = list1.to(torch.int64), list2.to(torch.int64)

    return list1, list2

# multi-head attention layer
class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        # mask = mask[:, :, :query.size(1)]         如果需要mask则使用，否则 # 
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) + 1e-9)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    # p_attn = entmax15(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn



# get embedding
def get_conspan_matrix(span_list, rm_loop=False, max_len=None):
    '''
    span_list: [N,B,L]
    return span:[N,B,L,L]
    '''
    # [N,B,L]
    N, B, L = span_list.shape
    span = get_span_matrix_3D(span_list.contiguous().view(-1, L), rm_loop, max_len).contiguous().view(N, B, L, L)
    return span

def get_span_matrix_3D(span_list, rm_loop=False, max_len=None):
    # [N,L]
    origin_dim = len(span_list.shape)
    if origin_dim == 1:  # [L]
        span_list = span_list.unsqueeze(dim=0)
    N, L = span_list.shape
    if max_len is not None:
        L = min(L, max_len)
        span_list = span_list[:, :L]
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    span = span * (span.transpose(-1, -2) == span)
    if rm_loop:
        span = span * (~torch.eye(L).bool()).unsqueeze(dim=0).repeat(N, 1, 1)
        span = span.squeeze(dim=0) if origin_dim == 1 else span  # [N,L,L]
    return span

def get_embedding(vocab, opt):
    graph_emb=0

    if 'laptop' in opt.dataset:
        graph_file = './embeddings/entity_embeddings_analogy_400.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_analogy.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_analogy_bert.pkl' % opt.dataset
        # graph_pkl = 'embeddings/%s_graph_analogy_roberta.pkl' % ds_name
    elif 'restaurant' in opt.dataset:
        graph_file = './embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_dismult_bert.pkl' % opt.dataset
        # graph_pkl = 'embeddings/%s_graph_dismult_roberta.pkl' % ds_name
    elif 'twitter' in opt.dataset:
        graph_file = './embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_dismult_bert.pkl' % opt.dataset

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



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

def ids2ori_adj(ori_tag, sent_len, head):
    adj = []
    # print(sent_len)
    for b in range(ori_tag.size()[0]):
        ret = np.ones((sent_len, sent_len), dtype='float32')
        fro_list = head[b]
        for i in range(len(fro_list) - 1):
            to = i + 1
            fro = fro_list[i]
            ret[fro][to] = ori_tag[b][i]
            ret[to][fro] =ori_tag[b][i]
        adj.append(ret)

    return adj

def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i in range(batch_size):
        sequence_mask[i, :lengths[i]] = True

    return sequence_mask.cuda()

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        self.RNN.flatten_parameters()
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM': 
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else: 
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)
    
class HFfusion(nn.Module):
    def __init__(self, opt, d_bert):
        super(HFfusion, self).__init__()
        self.opt = opt
        self.d_bert = d_bert
        self.hidden_dim = d_bert
        
        # Knowledge graph dimension projection (handle different input dimensions)
        # Based on the error, knowledge features seem to have different dimensions
        self.knowledge_proj = nn.Linear(2 * opt.lstm_dim + opt.dim_k, d_bert)
        
        # Local-level fusion components
        self.local_attention = nn.MultiheadAttention(d_bert, num_heads=8, dropout=0.1, batch_first=True)
        self.local_norm = nn.LayerNorm(d_bert)
        
        # Intermediate-level fusion components
        self.inter_cross_attn = nn.MultiheadAttention(d_bert, num_heads=8, dropout=0.1, batch_first=True)
        self.inter_norm = nn.LayerNorm(d_bert)
        
        # Global-level fusion components  
        self.global_cross_attn = nn.MultiheadAttention(d_bert, num_heads=4, dropout=0.1, batch_first=True)
        self.global_norm = nn.LayerNorm(d_bert)
        
        # Gating mechanisms for adaptive fusion
        self.syntactic_gate = nn.Sequential(
            nn.Linear(d_bert * 2, d_bert),
            nn.Sigmoid()
        )
        
        self.semantic_gate = nn.Sequential(
            nn.Linear(d_bert * 2, d_bert), 
            nn.Sigmoid()
        )
        
        self.knowledge_gate = nn.Sequential(
            nn.Linear(d_bert * 2, d_bert),
            nn.Sigmoid()
        )
        
        # Progressive fusion layers
        self.fusion_layer1 = nn.Sequential(
            nn.Linear(d_bert * 3, d_bert),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_bert)
        )
        
        self.fusion_layer2 = nn.Sequential(
            nn.Linear(d_bert * 2, d_bert),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.LayerNorm(d_bert)
        )
        
        self.final_fusion = nn.Sequential(
            nn.Linear(d_bert * 2, d_bert),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_bert)
        )
        
        # Feature enhancement modules
        self.feature_enhancer = nn.Sequential(
            nn.Linear(d_bert, d_bert),
            nn.GELU(),
            nn.Linear(d_bert, d_bert),
            nn.Dropout(0.1)
        )
        
        # Hierarchical attention weights
        self.level_weights = nn.Parameter(torch.ones(3))  # Local, Intermediate, Global
        
        # Cross-modal interaction
        self.cross_modal_proj = nn.Linear(d_bert, d_bert // 2)
        self.modal_interaction = nn.MultiheadAttention(d_bert // 2, num_heads=4, batch_first=True)
        
    def forward(self, bert_enc, graph_con, graph_dep, graph_seman, graph_amr, graph_know):
        batch_size = bert_enc.size(0)
        
        # Project knowledge features to match BERT dimension
        if len(graph_know.shape) == 2 and graph_know.size(-1) != self.d_bert:
            graph_know = self.knowledge_proj(graph_know)
        
        # Check if ablation study is requested based on fusion_condition
        if hasattr(self.opt, 'fusion_condition') and self.opt.fusion_condition != 'HF':
            # Ablation study with 7 specific combinations
            if self.opt.fusion_condition == "CD":
                # CD → Constituency + Dependency
                final_outputs = torch.cat((graph_con, graph_dep), dim=-1)
            elif self.opt.fusion_condition == "A":
                # A → AMR
                final_outputs = graph_amr
            elif self.opt.fusion_condition == "K":
                # K → Knowledge Graph
                final_outputs = graph_know
            elif self.opt.fusion_condition == "CD+A":
                # CD + A → Constituency + Dependency + AMR
                final_outputs = torch.cat((graph_con, graph_dep, graph_amr), dim=-1)
            elif self.opt.fusion_condition == "CD+K":
                # CD + K → Constituency + Dependency + Knowledge Graph
                final_outputs = torch.cat((graph_con, graph_dep, graph_know), dim=-1)
            elif self.opt.fusion_condition == "A+K":
                # A + K → AMR + Knowledge Graph
                final_outputs = torch.cat((graph_amr, graph_know), dim=-1)
            elif self.opt.fusion_condition == "CD+A+K":
                # CD + A + K → Constituency + Dependency + AMR + Knowledge Graph
                final_outputs = torch.cat((graph_con, graph_dep, graph_amr, graph_know), dim=-1)
            else:
                # Default to AMR if unknown fusion condition
                final_outputs = graph_amr
            
            return final_outputs
        
        # Original HF (Hierarchical Fusion) logic below
        # Reshape inputs for attention (add sequence dimension if needed)
        if len(bert_enc.shape) == 2:
            bert_enc = bert_enc.unsqueeze(1)  # [B, 1, D]
            graph_con = graph_con.unsqueeze(1)
            graph_dep = graph_dep.unsqueeze(1) 
            graph_seman = graph_seman.unsqueeze(1)
            graph_amr = graph_amr.unsqueeze(1)
            
        # Handle knowledge dimension properly
        if len(graph_know.shape) == 2:
            graph_know = graph_know.unsqueeze(1)
        
        # === Level 1: Local Syntactic Fusion ===
        # Fuse constituency and dependency representations
        syn_concat = torch.cat([graph_con, graph_dep], dim=-1)
        syn_gate = self.syntactic_gate(syn_concat)
        syn_fused = syn_gate * graph_con + (1 - syn_gate) * graph_dep
        
        # Local attention for syntactic features
        syn_attended, _ = self.local_attention(syn_fused, syn_fused, syn_fused)
        syn_local = self.local_norm(syn_fused + syn_attended)
        
        # === Level 2: Intermediate Semantic Integration ===
        # Integrate semantic and AMR representations
        sem_concat = torch.cat([graph_seman, graph_amr], dim=-1)
        sem_gate = self.semantic_gate(sem_concat)
        sem_fused = sem_gate * graph_seman + (1 - sem_gate) * graph_amr
        
        # Cross-attention between syntactic and semantic
        sem_cross_attn, _ = self.inter_cross_attn(sem_fused, syn_local, syn_local)
        sem_inter = self.inter_norm(sem_fused + sem_cross_attn)
        
        # Progressive fusion at intermediate level
        inter_features = torch.cat([syn_local, sem_inter, bert_enc], dim=-1)
        inter_fused = self.fusion_layer1(inter_features)
        
        # === Level 3: Global Knowledge Integration ===
        # Integrate knowledge graph information
        know_concat = torch.cat([inter_fused, graph_know], dim=-1)
        know_gate = self.knowledge_gate(know_concat)
        know_enhanced = know_gate * inter_fused + (1 - know_gate) * graph_know
        
        # Global cross-attention
        global_attn, _ = self.global_cross_attn(know_enhanced, inter_fused, inter_fused)
        global_features = self.global_norm(know_enhanced + global_attn)
        
        # === Cross-Modal Interaction ===
        # Project to lower dimension for efficient interaction
        bert_proj = self.cross_modal_proj(bert_enc)
        global_proj = self.cross_modal_proj(global_features)
        
        # Cross-modal attention
        cross_modal, _ = self.modal_interaction(bert_proj, global_proj, global_proj)
        cross_modal = torch.cat([cross_modal, global_proj], dim=-1)  # Restore dimension
        
        # === Final Hierarchical Fusion ===
        # Combine BERT and graph features
        bert_graph_concat = torch.cat([bert_enc, global_features], dim=-1)
        bert_graph_fused = self.fusion_layer2(bert_graph_concat)
        
        # Final fusion with cross-modal interaction
        final_concat = torch.cat([bert_graph_fused, cross_modal], dim=-1)
        final_output = self.final_fusion(final_concat)
        
        # Feature enhancement
        enhanced_output = self.feature_enhancer(final_output)
        final_output = final_output + enhanced_output  # Residual connection
        
        # Apply hierarchical level weights (learnable importance)
        level_weights = F.softmax(self.level_weights, dim=0)
        weighted_output = (level_weights[0] * syn_local + 
                          level_weights[1] * inter_fused + 
                          level_weights[2] * final_output)
        
        # Remove sequence dimension if added
        if weighted_output.size(1) == 1:
            weighted_output = weighted_output.squeeze(1)
            
        return weighted_output

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
    def __init__(self, emb_dim, con_layers, dep_layers, sem_layers, amr_layers = 9, gcn_dropout=0.1):
        super().__init__()
        self.con_layers = con_layers
        self.dep_layers = dep_layers
        self.sem_layers = sem_layers
        self.amr_layers = amr_layers

        self.emb_dim = emb_dim
        self.out_dim = emb_dim
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

    def forward(self, inputs, con_adj, dep_adj, seman_adj, amr_adj, tok_length):
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

        # print(amr_input)
        multi_loss = process_adj_matrices(dep_adj, con_adj[0].bool().float(), seman_adj, amr_adj, con_input, dep_input, amr_input, tok_length)
        # print(multi_loss)
        return con_input, dep_input, seman_input, amr_input, multi_loss
    
def process_adj_matrices(dep_adj, con_adj_new, seman_adj, amr_adj, con_input, dep_input, amr_input, tok_len):
    batch_size, max_length, _ = dep_adj.size()
    
    multi_viewdep_loss = 0
    multi_viewamr_loss = 0  # NEW: accumulate AMR-related loss

    for b in range(batch_size):
        length = int(tok_len[b])

        # Slice the tensors to the required length
        dep_adj_batch = dep_adj[b, :length, :length].to(dtype=torch.int)
        con_adj_new_batch = con_adj_new[b, :length, :length].to(dtype=torch.int)
        sem_adj_batch = seman_adj[b, :length, :length]
        amr_adj_batch = amr_adj[b, :length, :length].to(dtype=torch.int)  # AMR ADJ

        # caculate importance scores
        node_importance_scores = sem_adj_batch.mean(dim=1) + sem_adj_batch.max(dim=1).values
        k = int(math.log10(length) ** 2)

        top_k_node = torch.topk(node_importance_scores, k).indices.tolist()

        dep_non_zero_indices = torch.nonzero(dep_adj_batch)
        con_non_zero_indices = torch.nonzero(con_adj_new_batch)

        dep_edges_tuple = dep_non_zero_indices.t()
        con_edges_tuple = con_non_zero_indices.t()

        dep_1_start = dep_edges_tuple[0]
        dep_1_end = dep_edges_tuple[1]

        dep_3_start, dep_3_end = get_2nd_order_pairs(con_edges_tuple, dep_edges_tuple)

        multi_Dep_loss = 0

        for i in top_k_node:
            # DEP-CON
            Anchor_Dep = dep_input[b, i]
            AD_dep_view_node_index = dep_1_end[dep_1_start == i]
            AD_con_view_node_index = dep_3_end[dep_3_start == i]

            AD_dep_view_node = dep_input[b, AD_dep_view_node_index]
            AD_remaining_indices_inter = torch.nonzero(~AD_dep_view_node_index.new_full((length,), 0, dtype=torch.bool)).squeeze()
            AD_dep_view_node_remain = dep_input[b, AD_remaining_indices_inter]

            # inter-inter
            AD_con_view_node_index = torch.cat((AD_con_view_node_index, torch.tensor([i]).cuda()))
            AD_con_view_node = con_input[b, AD_con_view_node_index]
            AD_remaining_indices_intra = torch.nonzero(~AD_con_view_node_index.new_full((length,), 0, dtype=torch.bool)).squeeze()
            AD_con_view_node_remain = dep_input[b, AD_remaining_indices_intra]

            # total
            AD_P = torch.cat((AD_dep_view_node, AD_con_view_node), dim=0)
            AD_N = torch.cat((AD_dep_view_node_remain, AD_con_view_node_remain), dim=0)

            AD_loss = multi_margin_contrastive_loss(Anchor_Dep, AD_P, AD_N)

            multi_Dep_loss += AD_loss

            # ----------------------
            # AMR CONTRASTIVE LOSS
            # ----------------------
            # Anchor_AMR = amr_input[b, i]
            # amr_view_node_index = (amr_adj_batch[i] > 0).nonzero(as_tuple=True)[0]
            # # Remove self-loop if you don't want the anchor as its own positive
            # amr_view_node_index = amr_view_node_index[amr_view_node_index != i]
            # AMR_view_node = amr_input[b, amr_view_node_index] if amr_view_node_index.numel() > 0 else Anchor_AMR.unsqueeze(0)

            # # Negative: all nodes NOT in amr_view_node_index
            # all_indices = torch.arange(length, device=amr_input.device)
            # neg_indices = all_indices[~torch.isin(all_indices, amr_view_node_index)]
            # AMR_view_node_neg = amr_input[b, neg_indices] if neg_indices.numel() > 0 else Anchor_AMR.unsqueeze(0)

            # # Compute contrastive loss for AMR
            # AMR_loss = multi_margin_contrastive_loss(Anchor_AMR, AMR_view_node, AMR_view_node_neg)
            # multi_viewamr_loss += AMR_loss
            # print(f"Batch {b}, Node {i}: AMR Loss: {AMR_loss.item()}, Dep Loss: {AD_loss.item()}")
            # breakpoint()

        # multi_viewdep_loss += multi_Dep_loss + multi_viewamr_loss
        multi_viewdep_loss += multi_Dep_loss

    return multi_viewdep_loss

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
        graph_file = 'embeddings/entity_embeddings_analogy_400.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_analogy.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_analogy_bert.pkl' % opt.dataset
        # graph_pkl = 'embeddings/%s_graph_analogy_roberta.pkl' % ds_name
    elif 'restaurant' in opt.dataset:
        graph_file = 'embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_dismult_bert.pkl' % opt.dataset
        # graph_pkl = 'embeddings/%s_graph_dismult_roberta.pkl' % ds_name
    elif 'twitter' in opt.dataset:
        graph_file = 'embeddings/entity_embeddings_distmult_200.txt'
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

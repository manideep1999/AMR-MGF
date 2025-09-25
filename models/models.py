'''
Description: 
version: 
'''
from logging import root
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from .layer import *


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EHFBClassifier(nn.Module):
    def __init__(self, bert, embeddings, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, embeddings, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        
        self.v_linear = nn.Linear(in_features=opt.bert_dim,
                                  out_features=1,
                                  bias=False)

    def forward(self, inputs):
        tok_length, bert_length, bert_sequence, bert_segments_ids, word_mapback, map_AS, aspect_token, aspect_mask, src_mask, dep_spans, con_spans, amr_spans = inputs
        # print("tok_length, bert_length, bert_sequence.shape, bert_segments_ids.shape, word_mapback.shape, map_AS.shape, aspect_token.shape, aspect_mask.shape, src_mask.shape, dep_spans.shape, con_spans.shape" , len(tok_length), len(bert_length), bert_sequence.shape, bert_segments_ids.shape, word_mapback.shape, map_AS.shape, aspect_token.shape, aspect_mask.shape, src_mask.shape, dep_spans.shape, con_spans.shape)
        gcn_model_inputs = (tok_length[map_AS], bert_length[map_AS], bert_sequence, bert_segments_ids, word_mapback[map_AS], aspect_token, aspect_mask, src_mask, dep_spans[map_AS], con_spans, amr_spans[map_AS])
        outputs = self.gcn_model(gcn_model_inputs) 
        logits = outputs

        return logits


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, embeddings, opt):
        super().__init__()
        self.opt = opt
        # gcn layer
        self.gcn = GCNBert(bert, embeddings, opt, opt.num_layers)

        # Hierarchical Fusion Module
        self.HFfusion = HFfusion(opt, opt.bert_dim)
        
        # Determine classifier input dimension based on fusion_condition
        classifier_dim = self._get_classifier_input_dim(opt)
        self.classifier = nn.Linear(classifier_dim, opt.polarities_dim)

    def _get_classifier_input_dim(self, opt):
        """Calculate the input dimension for classifier based on fusion_condition"""
        bert_dim = opt.bert_dim
        
        if not hasattr(opt, 'fusion_condition') or opt.fusion_condition == 'HF':
            return bert_dim  # Original HF returns bert_dim
        elif opt.fusion_condition == "CD" or opt.fusion_condition == "A+K":
            # CD → Constituency + Dependency or A + K → AMR + Knowledge Graph
            return bert_dim * 2
        elif opt.fusion_condition == "A" or  opt.fusion_condition == "K":
            # A → AMR or K → Knowledge Graph
            return bert_dim
        elif opt.fusion_condition == "CD+A" or opt.fusion_condition == "CD+K":
            # CD + A → Constituency + Dependency + AMR or CD + K → Constituency + Dependency + Knowledge Graph
            return bert_dim * 3
        elif opt.fusion_condition == "CD+A+K":
            # CD + A + K → Constituency + Dependency + AMR + Knowledge Graph
            return bert_dim * 4
        else:
            return bert_dim  # Default to BERT dimension

    def forward(self, inputs):
        tok_length, bert_length, bert_sequence, bert_segments_ids, word_mapback, aspect_token, aspect_mask, src_mask, dep_spans, con_spans, amr_spans = inputs           # unpack inputs
        con_output, dep_output, seman_output, amr_output, know_output, gcn_inputs, pooled_output, multi_loss = self.gcn(inputs)  
        # print(multi_loss)
        aspect_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)  # mask for h

        # bert
        bert_enc_outputs = (gcn_inputs * aspect_mask).sum(dim=1) / aspect_wn

        # graph 
        graph_con_outputs = (con_output * aspect_mask).sum(dim=1) / aspect_wn
        graph_dep_outputs = (dep_output * aspect_mask).sum(dim=1) / aspect_wn
        graph_seman_outputs = (seman_output * aspect_mask).sum(dim=1) / aspect_wn
        graph_amr_outputs = (amr_output * aspect_mask).sum(dim=1) / aspect_wn
        graph_know_outputs = know_output

        final_output = self.HFfusion(bert_enc_outputs, graph_con_outputs, graph_dep_outputs, graph_seman_outputs, graph_amr_outputs, graph_know_outputs)

        logits = self.classifier(final_output)

        return logits, multi_loss
        
class GCNBert(nn.Module):
    def __init__(self, bert, embeddings, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # gcn layer
        self.GCNEncoder = GCNEncoder(opt.bert_dim, opt.max_num_spans, opt.dep_layers, opt.sem_layers, opt.amr_layers, kg_dim=opt.dim_k)
        self.attention_heads = opt.attention_heads
        self.attn = MultiHeadAttention(self.attention_heads, self.bert_dim) 
        self.dense = nn.Linear(opt.bert_dim, opt.hidden_dim)

        # knowledge graph
        self.graph_embeddings = embeddings
        self.knowledge_embed = nn.Embedding.from_pretrained(torch.from_numpy(self.graph_embeddings).float().cuda(), freeze=True)
        self.text_lstm = DynamicLSTM(opt.dim_w, opt.lstm_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.aspect_lstm = DynamicLSTM(opt.dim_w, opt.lstm_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.text_embed_dropout = nn.Dropout(opt.dropout_rate)
    
    def forward(self,inputs):
        tok_length, bert_length, bert_sequence, bert_segments_ids, word_mapback, aspect_token, aspect_mask, src_mask, dep_spans, con_spans, amr_spans = inputs      
        # print("tok_length, bert_length, bert_sequence.shape, bert_segments_ids.shape, word_mapback.shape, aspect_token.shape, aspect_mask.shape, src_mask.shape, dep_spans.shape, con_spans.shape, amr_spans.shape" , tok_length.shape, bert_length.shape, bert_sequence.shape, bert_segments_ids.shape, word_mapback.shape, aspect_token.shape, aspect_mask.shape, src_mask.shape, dep_spans.shape, con_spans.shape, amr_spans.shape)
        bert_outputs = self.bert(bert_sequence, token_type_ids=bert_segments_ids)   # 如果有padding最好加上attention_mask，否则不需要加上
        sequence_output, pooled_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        sequence_output = self.layernorm(sequence_output)
        bert_out = self.bert_drop(sequence_output)

        # remove [CLS], aspect and [SEP]
        bert_seq_indi = sequence_mask(bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()
        # print("bert_out.shape, bert_seq_indi.shape", bert_out.shape, bert_seq_indi.shape)
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))

        # average
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        gcn_inputs = bert_out / wnt.unsqueeze(dim=-1)  

        # 1、constituent and dependent fusion
        con_matirx = get_conspan_matrix(con_spans.transpose(0, 1))

        # 2、semantic
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask.unsqueeze(-2))
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads    

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).to(self.opt.device)
        seman_matrix = src_mask.unsqueeze(-1) * adj_ag  

        # 3、knowledge graph embedding
        knowledge_embedding = self.knowledge_embed(bert_sequence)  
        knowledge_embedding = self.text_embed_dropout(knowledge_embedding)
        knowledge_embedding = knowledge_embedding[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()
        knowledge_embedding = torch.bmm(word_mapback_one_hot.float(), knowledge_embedding)
        text_out, (_, _) = self.text_lstm(bert_out, tok_length.cpu())
        text_out = self.text_embed_dropout(text_out)

        aspect_knowledge_embedding = self.knowledge_embed(aspect_token)
        aspect_knowledge_embedding = self.text_embed_dropout(aspect_knowledge_embedding)
        aspect_token_embeding = self.bert(aspect_token).last_hidden_state
        aspect_token_embeding = self.text_embed_dropout(aspect_token_embeding)
        aspect_len = torch.sum(aspect_token != 0, dim=-1)
        aspect_out, (_, _) = self.aspect_lstm(aspect_token_embeding, aspect_len.cpu())
        aspect_out = self.text_embed_dropout(aspect_out)

        text_knowledge = torch.cat((knowledge_embedding, text_out), dim=-1)
        aspect_knowledge = torch.cat((aspect_knowledge_embedding, aspect_out), dim=-1)
        knowledge_score = torch.bmm(aspect_knowledge, text_knowledge.transpose(1, 2))
        knowledge_score = F.softmax(knowledge_score, dim=-1)
        knowledge_out = torch.bmm(knowledge_score, text_knowledge)
        know_output = F.max_pool1d(knowledge_out.transpose(1, 2), knowledge_out.shape[1]).squeeze(2) 

        # GCN Update Module
        dep_matrix = dep_spans
        amr_matrix = amr_spans

        # Pass knowledge embeddings to GCNEncoder for KG loss computation
        con_out, dep_out, seman_out, amr_output, multi_loss = self.GCNEncoder(gcn_inputs, con_matirx, dep_matrix, seman_matrix, amr_matrix, tok_length, knowledge_embedding, self.opt)
            # print(amr_output)
        return con_out, dep_out, seman_out, amr_output, know_output, gcn_inputs, pooled_output, multi_loss











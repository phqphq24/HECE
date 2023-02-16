import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn.utils.rnn import pad_sequence
from configs import DEVICE
from transformers import BertModel, BertConfig
from gnn_layer import GraphAttentionLayer
from utils import *
from attention import *


class Feature_Extractor(nn.Module):
    def __init__(self, configs):
        super(Feature_Extractor, self).__init__()
        self.config = BertConfig.from_pretrained(configs.bert_cache_path)
        self.Bert = BertModel.from_pretrained(configs.bert_cache_path)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b):
        output = self.Bert(input_ids=bert_token_b.to(DEVICE),
                           attention_mask=bert_masks_b.to(DEVICE),
                           token_type_ids=bert_segment_b.to(DEVICE))
        return output[0]


class Utt_Classify(nn.Module):
    def __init__(self, configs):
        super(Utt_Classify, self).__init__()
        self.gnn = GraphNN(configs)
        self.feat_dim = int(configs.gnn_dims.strip().split(',')[-1]) * int(configs.att_heads.strip().split(',')[-1])
        self.out_e = nn.Linear(self.feat_dim+configs.pos_dim+configs.emo_dim, 1)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, hidden_state, bert_clause_idx_b, emotion_b, relative_pos_b, adj_b, conv_len_b, emotion_embedding, emotion_list, pos_embedding):
        conv_sents_h = self.batched_index_select(hidden_state, bert_clause_idx_b)
        conv_sents_h = self.gnn(conv_sents_h, conv_len_b, adj_b)   
        conv_emo_utt = []
        for i in range(conv_sents_h.size(0)):
            conv_emo_utt.append(conv_sents_h[i][conv_len_b[i]-1])
        conv_emo_utt = torch.stack(conv_emo_utt, dim=0)
        conv_len = conv_sents_h.size(1)
        emotion_id = []
        for emotion in emotion_b:
            emotion_id.append(emotion_list[emotion])
        emotion_id = torch.LongTensor(emotion_id).to(DEVICE)
        emotion_embed = emotion_embedding(emotion_id)  
        emotion_embed_ = emotion_embed.unsqueeze(1).expand(-1, conv_len, -1) 

        relative_pos_b = relative_pos_b.to(DEVICE)
        pos_embed = pos_embedding(relative_pos_b) 
        kernel_rel_pos_b = self.kernel_generator(relative_pos_b)
        pos_embed = torch.matmul(kernel_rel_pos_b, pos_embed)
        conv_sents_h_ = torch.cat([emotion_embed_, conv_sents_h, pos_embed], dim=-1) 
        utt_prob = self.out_e(conv_sents_h_).squeeze(2)
        return utt_prob, conv_emo_utt 


    def batched_index_select(self, hidden_state, bert_clause_idx_b):
        bert_clause_idx_b = bert_clause_idx_b.to(DEVICE)
        dummy = bert_clause_idx_b.unsqueeze(2).expand(bert_clause_idx_b.size(0), bert_clause_idx_b.size(1), self.feat_dim)
        conv_sents_h = hidden_state.gather(1, dummy)
        return conv_sents_h

    def kernel_generator(self, relative_pos_b):
        batch, seq_len = relative_pos_b.size()
        kernel_rel_pos_b = []
        for i in range(batch):
            relative_pos = relative_pos_b[i]
            relative_pos_ = relative_pos.type(torch.FloatTensor).to(DEVICE)
            kernel_left = torch.cat([relative_pos_.reshape(-1, 1)] * seq_len, dim=1)
            kernel = kernel_left - kernel_left.transpose(0, 1)
            kernel_rel_pos_b.append(torch.exp(-(torch.pow(kernel, 2))))
        return torch.stack(kernel_rel_pos_b, dim=0) 


class Span_Classify(nn.Module):
    def __init__(self, configs):
        super(Span_Classify, self).__init__()
        self.feat_dim = configs.feat_dim
        self.emo_dim = configs.emo_dim
        self.pos_dim = configs.pos_dim
        self.word_token = configs.word_token
        self.attention = MultiHeadAttention(in_features_1=self.feat_dim, in_features_2=self.feat_dim, head_num=12)
        self.out_span = nn.Linear(configs.feat_dim*2, 1)
        self.dropout = nn.Dropout(configs.dropout)
    
    def forward(self, hidden_state, bert_word_idx_b, emotion_b, conv_clause_len_b, conv_len_b, emotion_embedding, emotion_list, utt_prob, conv_emo_utt, pos_embedding, relative_pos_b):
        conv_words_h = self.batched_index_select(hidden_state, bert_word_idx_b)  
        emo_utt_words = self.get_emo_words(conv_words_h, conv_clause_len_b)
        query = conv_words_h
        key = emo_utt_words
        value = emo_utt_words
        res, new_q, word_attn = self.attention(query, key, value)
        new_conv_words_h = res + new_q

        conv_words_h = torch.cat([conv_words_h, new_conv_words_h], dim=-1)

        max_conv_word_len = conv_words_h.size(1)
        utt_pred_label = self.build_utt_pred_mask(utt_prob, conv_clause_len_b, conv_len_b, max_conv_word_len) 
        utt_pred_label = utt_pred_label.to(DEVICE)

        span_prob = self.out_span(conv_words_h).squeeze(-1)
        return span_prob, utt_pred_label

    def get_emo_words(self, conv_words_h, conv_clause_len_b):
        emo_utt_words_b = []
        batch = conv_words_h.size(0)
        max_emo_utt_len = max([conv_clause_len[-1] for conv_clause_len in conv_clause_len_b])
        for i in range(batch):
            temp_conv_words = conv_words_h[i]
            temp_conv_clause_len = conv_clause_len_b[i]
            emo_utt_len_start = sum(temp_conv_clause_len[:-1])
            emo_utt_words = temp_conv_words[emo_utt_len_start:emo_utt_len_start+temp_conv_clause_len[-1]]
            emo_utt_words = torch.cat([emo_utt_words, torch.zeros(max_emo_utt_len-temp_conv_clause_len[-1], self.feat_dim).float().to(DEVICE)], dim=0)
            emo_utt_words_b.append(emo_utt_words)
        return torch.stack(emo_utt_words_b, dim=0)

    def batched_index_select(self, hidden_state, bert_word_idx_b):
        bert_word_idx_b = bert_word_idx_b.to(DEVICE)
        dummy = bert_word_idx_b.unsqueeze(2).expand(bert_word_idx_b.size(0), bert_word_idx_b.size(1), self.feat_dim)
        conv_words_h = hidden_state.gather(1, dummy)
        return conv_words_h


    def build_utt_pred_mask(self, utt_prob, conv_clause_len_b, conv_len_b, max_conv_word_len):
        batch = len(conv_clause_len_b)
        utt_pred_mask = []
        for i in range(batch):
            temp_utt_pred_mask = []
            temp_conv_len = conv_len_b[i]
            conv_clause_len = conv_clause_len_b[i]
            for j in range(temp_conv_len):
                if logistic(utt_prob[i][j]) > 0.5
                    temp_utt_pred_mask += [1] * conv_clause_len[j]
                else:
                    temp_utt_pred_mask += [0] * conv_clause_len[j]
            temp_len = len(temp_utt_pred_mask)
            temp_utt_pred_mask += [0] * (max_conv_word_len - temp_len)
            utt_pred_mask.append(temp_utt_pred_mask)
        return torch.LongTensor(utt_pred_mask)



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.feature_extractor = Feature_Extractor(configs)
        self.utt_model = Utt_Classify(configs)
        self.span_model = Span_Classify(configs)
        self.emotion_list = {'anger': 0, 'disgust': 1, 'sadness': 2, 'excited': 3, 'fear': 4, 'surprise': 5, 'happiness': 6}
        self.emotion_embedding = nn.Embedding(7, configs.emo_dim)
        nn.init.xavier_uniform_(self.emotion_embedding.weight)
        self.pos_embedding = nn.Embedding(45, configs.pos_dim)
        nn.init.xavier_uniform_(self.pos_embedding.weight)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_idx_b,
                bert_word_idx_b, emotion_b, relative_pos_b, adj_b, conv_len_b, conv_clause_len_b, task='both'):
        hidden_state = self.feature_extractor(bert_token_b, bert_segment_b, bert_masks_b)
        utt_prob, conv_emo_utt = self.utt_model(hidden_state, bert_clause_idx_b, emotion_b, relative_pos_b, adj_b, conv_len_b, self.emotion_embedding, self.emotion_list, self.pos_embedding)
        span_prob, utt_pred_label = self.span_model(hidden_state, bert_word_idx_b, emotion_b, conv_clause_len_b, conv_len_b, self.emotion_embedding, self.emotion_list, utt_prob, conv_emo_utt, self.pos_embedding, relative_pos_b)
        return utt_prob, span_prob, utt_pred_label


    def utt_loss(self, utt_prob, conv_utt_label_b, conv_utt_mask_b):
        utt_mask = conv_utt_mask_b.ge(0.1).to(DEVICE)
        utt_label = conv_utt_label_b.float().to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        utt_prob = utt_prob.masked_select(utt_mask)
        utt_label = utt_label.masked_select(utt_mask)
        loss_utt = criterion(utt_prob, utt_label)
        return loss_utt

    def span_loss(self, span_prob, conv_span_label_b, conv_span_mask_b):
        span_mask = conv_span_mask_b.ge(0.1).to(DEVICE)
        span_label = conv_span_label_b.float().to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        span_prob = span_prob.masked_select(span_mask)
        span_label = span_label.masked_select(span_mask)
        loss_span = criterion(span_prob, span_label)
        return loss_span


class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h, attn = gnn_layer(doc_sents_h, adj)

        return doc_sents_h, attn

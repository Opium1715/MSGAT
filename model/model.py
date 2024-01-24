import math

import torch
import torch.nn.functional as F
from entmax import entmax_bisect
from torch import nn


class GGNN(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.gnn_cell = nn.GRUCell(input_size=4 * self.emb_size,
                                   hidden_size=2 * self.emb_size,
                                   bias=True)
        self.linear_H_in = nn.Linear(self.emb_size * 2, self.emb_size * 2,
                                     bias=False)
        self.linear_H_out = nn.Linear(self.emb_size * 2, self.emb_size * 2,
                                      bias=False)
        self.bias_in = nn.Parameter(torch.Tensor(self.emb_size * 2))
        self.bias_out = nn.Parameter(torch.Tensor(self.emb_size * 2))

    def forward(self, A, X):
        batch_size = A.shape[0]
        length = A.shape[1]
        A_in, A_out = torch.chunk(A, chunks=2, dim=-1)
        adj_in = torch.matmul(A_in, self.linear_H_in(X)) + self.bias_in
        adj_out = torch.matmul(A_out, self.linear_H_out(X)) + self.bias_out
        inputs = torch.concat([adj_in, adj_out], -1)
        inputs = inputs.contiguous().view([batch_size * length, -1])
        result = self.gnn_cell(inputs, X.contiguous().view([batch_size * length, -1])).contiguous().view(
            [batch_size, length, -1])
        return result


class S_ATT(nn.Module):
    def __init__(self, emb_size, drop_out, head, *args, **kwargs):  # multi_head
        super().__init__(*args, **kwargs)
        self.head = head
        self.emb_size = emb_size
        self.attn_head_size = int(self.emb_size * 2 / self.head)
        self.linear_q = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.drop_out = drop_out
        self.scale = math.sqrt(self.emb_size * 2 / head)
        self.ffn = nn.Sequential(nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                                 nn.Dropout(self.drop_out))
        self.layer_norm = nn.LayerNorm(2 * self.emb_size, eps=1e-12)
        self.drop_q = nn.Dropout(self.drop_out)
        self.drop_attn = nn.Dropout(self.drop_out)
        self.linear_alpha_s = nn.Linear(int(self.emb_size * 2 / self.head), 1)

    def forward(self, X_target_plus, mask):
        batch_size = X_target_plus.shape[0]
        length = X_target_plus.shape[1]
        q = self.drop_q(torch.relu(self.linear_q(X_target_plus)))  # q drop
        k = X_target_plus
        v = X_target_plus
        # multi-head
        multi_head_q = q.view(batch_size, length, self.head, self.attn_head_size).permute(0, 2, 1, 3).contiguous()
        multi_head_k = k.view(batch_size, length, self.head, self.attn_head_size).permute(0, 2, 3, 1).contiguous()
        multi_head_v = v.view(batch_size, length, self.head, self.attn_head_size).permute(0, 2, 1, 3).contiguous()
        attention_score = torch.matmul(multi_head_q, multi_head_k) / self.scale
        # mask
        attention_mask = torch.concat([mask, torch.zeros([batch_size, 1], dtype=torch.float32, device='cuda')], dim=-1)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand_as(attention_score)
        attention_score = attention_score.masked_fill(attention_mask == 0, -torch.inf)
        # alpha
        alpha = torch.sigmoid(self.linear_alpha_s(multi_head_q[:, :, -1, :])) + 1  # [1, 2]  注意用的是线性变换后的q中拆出来的
        alpha = torch.clip(alpha, min=1 + 1e-5).unsqueeze(-1).expand(-1, -1, X_target_plus.shape[1],
                                                                     -1)  # entmax不能实现alpha=1时的softmax,所以给个裁切
        attention_score = entmax_bisect(attention_score, alpha=alpha, dim=-1)
        attention_score = self.drop_attn(attention_score)
        attention_result = torch.matmul(attention_score, multi_head_v)
        attention_result = attention_result.permute(0, 2, 1, 3).contiguous().view(batch_size, length, self.emb_size * 2)
        C_hat = self.layer_norm(self.ffn(attention_result) + attention_result)
        target_emb = C_hat[:, -1, :]
        C = C_hat[:, :-1, :]
        return C, target_emb


class G_ATT(nn.Module):
    def __init__(self, emb_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.linear_Wg1 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wg2 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wg0 = nn.Linear(self.emb_size * 2, 1, bias=False)
        self.bias_alpha = nn.Parameter(torch.Tensor(self.emb_size * 2))

    def forward(self, C_target, C, X, alpha, mask):
        q = C_target.unsqueeze(1)
        k = C
        v = C  # 注意这里不一样
        attention_score = self.linear_Wg0(torch.relu(self.linear_Wg1(k) + self.linear_Wg2(q) + self.bias_alpha))
        # mask
        attention_score = attention_score.masked_fill(mask.unsqueeze(-1) == 0, -torch.inf)
        attention_score = entmax_bisect(attention_score, alpha=alpha, dim=1)
        attention_result = torch.matmul(attention_score.transpose(1, 2), v)
        attention_result = torch.sum(torch.matmul(attention_result, X), dim=1)
        return attention_result


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

    def forward(self, A_r, D, R):  # D要不要求逆
        return torch.matmul(torch.matmul(D, A_r), R)


class SparseTargetAttention(nn.Module):
    def __init__(self, emb_size):
        super(SparseTargetAttention, self).__init__()
        self.emb_size = emb_size
        self.linear_Wr1 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wr2 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_Wr3 = nn.Linear(self.emb_size * 2, 1, bias=False)
        self.bias_r = nn.Parameter(torch.Tensor(self.emb_size * 2))
        self.linear_Wf = nn.Linear(self.emb_size * 2, self.emb_size)

    def forward(self, R, alpha, target):
        q = self.linear_Wf(target)
        k = R
        v = R
        attention_score = self.linear_Wr3(torch.relu(self.linear_Wr1(k) + self.linear_Wr2(q) + self.bias_r))
        attention_score = entmax_bisect(attention_score, alpha)  # 没有mask?
        attention_result = torch.matmul(attention_score.transpose(1, 2), v)
        # 某种正则化？
        attention_result = torch.selu(attention_result).squeeze()
        attention_result = attention_result / torch.norm(attention_result, dim=-1).unsqueeze(1)

        return attention_result


class SimilarIntent(nn.Module):
    def __init__(self, theta, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta
        self.top_k = top_k
        self.dropout = nn.Dropout(0.4)

    def forward(self, h):
        Sim = F.cosine_similarity(h.unsqueeze(0), h.unsqueeze(1), dim=-1)  # 1 512 100   512 1 100  --> 512 512
        # if h.shape[0] < self.top_k:  # 最后一批次可能小过下限
        #     self.top_k = h.shape[0]
        sim_topK, indices_topK = torch.topk(Sim, k=self.top_k, dim=-1)
        beta = torch.softmax(self.theta * sim_topK, dim=-1)  # 512 5
        h_topK = h[indices_topK]  # 512 5 100
        beta = beta.unsqueeze(-1)
        h_sim = self.dropout(torch.sum(beta * h_topK, 1))
        return h_sim


class MSGAT(nn.Module):
    def __init__(self, emb_size, item_num, max_len, drop_out, gamma, theta, top_k, omega, head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.item_num = item_num
        self.max_len = max_len
        self.dropout = drop_out
        self.gamma = gamma  # target_enhance
        self.drop_out = nn.Dropout(drop_out)
        self.ggnn_layer = GGNN(self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num + 1, self.emb_size, padding_idx=0, max_norm=1.5)
        self.position_embedding = nn.Embedding(self.max_len, self.emb_size, max_norm=1.5)

        self.linear_alpha_g = nn.Linear(self.emb_size * 2, 1)
        self.sa_layer = S_ATT(self.emb_size, self.dropout, head)
        # self.target_enhance_layer = TargetEnhance(self.emb_size, self.gamma)
        self.ga_layer = G_ATT(self.emb_size)
        self.linear_wc = nn.Linear(self.emb_size * 4, self.emb_size)
        self.similar_intent_layer = SimilarIntent(theta, top_k)
        self.omega = omega
        self.linear_w1 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.linear_w2 = nn.Linear(self.emb_size * 2, self.emb_size * 2, bias=False)
        self.gcn_layer = GCN()
        self.linear_alpha_r = nn.Linear(self.emb_size * 2, self.emb_size * 2)
        self.sta_layer = SparseTargetAttention(self.emb_size)

    def forward(self, alias_index, A, item, A_r, D, mask):
        batch_size = item.shape[0]
        item_len = item.shape[1]
        seq_len = mask.shape[1]
        item_emb = self.item_embedding(item)
        pos_emb = self.position_embedding(torch.arange(item_len, device='cuda', dtype=torch.int64)).unsqueeze(0).repeat(
            batch_size, 1, 1)
        X = torch.cat([item_emb, pos_emb], dim=-1)
        X = self.ggnn_layer(A, X)

        # rebuild to seq
        session = X[torch.arange(batch_size, dtype=torch.int64, device='cuda').unsqueeze(1), alias_index,
                  :]  # broadcast

        # soft attention
        last_index = torch.sum(mask, dim=-1).to(torch.int64) - 1
        last_click_item = session[torch.arange(batch_size, dtype=torch.int64, device='cuda'), last_index, :]
        H_g = torch.sum(torch.sigmoid(
            self.linear_w1(last_click_item.unsqueeze(1).repeat(1, seq_len, 1) * mask.unsqueeze(-1)) + self.linear_w2(
                session)), dim=1)

        # self attention
        X_target_plus = torch.concat(
            [session, torch.zeros((batch_size, 1, self.emb_size * 2), dtype=torch.float32, device='cuda')], 1)
        C, target = self.sa_layer(X_target_plus, mask)

        # global-attention
        alpha = torch.sigmoid(self.linear_alpha_g(target)) + 1  # [1, 2]
        alpha = torch.clip(alpha, 1 + 1e-5).unsqueeze(1)
        H_g_att = self.ga_layer(target, C, H_g, alpha, mask)

        # decoder
        H_c = self.drop_out(torch.selu(self.linear_wc(torch.concat([H_g_att, target], dim=-1))))
        H_c = H_c / torch.norm(H_c, dim=-1, keepdim=True)  # 正则

        # GCN
        R_init = torch.sum(item_emb, dim=1)
        R = self.gcn_layer(A_r, D, R_init)

        # SparseTargetAttention
        alpha = torch.sigmoid(self.linear_alpha_r(target)) + 1
        H_relation_att = self.sta_layer(R, alpha, target)

        # similarity
        H_sim = H_c + H_relation_att
        session_neighbor = self.similar_intent_layer(H_sim)

        # predict
        session_final = H_c + session_neighbor
        item_norm = F.normalize(
            self.item_embedding(torch.arange(1, self.item_num + 1, device='cuda', dtype=torch.int64)), dim=-1)
        score = torch.matmul(session_final, item_norm.transpose(1, 0))

        return score

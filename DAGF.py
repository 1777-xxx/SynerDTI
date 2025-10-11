import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DAGFusion(nn.Module):
    def __init__(self, feat_dim=128, h_dim=256, h_out=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.h_dim = h_dim

        # 1. 特征投影（带残差适配）
        self.feat_proj_v = nn.Linear(feat_dim, h_dim)
        self.feat_proj_q = nn.Linear(feat_dim, h_dim)
        self.feat_proj_res = nn.Linear(h_dim, h_dim)

        # 2. 全局特征分支（对应示意图里的 X、P 分支）
        self.global_branch_v = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Tanh()
        )
        self.global_branch_q = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Tanh()
        )

        # 3. 融合模块
        self.fusion_Wb = nn.Linear(h_dim * 2, h_dim * 2)
        self.fusion_C = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim * 4),
            nn.Tanh(),
            nn.Linear(h_dim * 4, h_dim * 2),
            nn.LayerNorm(h_dim * 2)
        )
        self.fusion_res = nn.Linear(h_dim * 2, h_dim * 2)


        # 分支 v 的注意力
        self.attn_branch_v = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1),
            nn.Softmax(dim=1)  # 生成 v 分支的注意力权重
        )
        # 分支 q 的注意力
        self.attn_branch_q = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1),
            nn.Softmax(dim=1)  # 生成 q 分支的注意力权重
        )

        self.main_output = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),  # 1280 → 256
            nn.GELU(),#ReLU
            nn.Linear(h_dim, 256)
        )

        self.main_output_res = nn.Linear(256, 256)  # 主输出残差适配

        # 6. 位置编码（保持原有）
        self.pos_encoder = PositionalEncoding2D(feat_dim)

    def forward(self, v_feat, q_feat):
        # --------------- 1. 特征投影 + 残差 ---------------
        v_proj = self.feat_proj_v(v_feat)
        q_proj = self.feat_proj_q(q_feat)

        # --------------- 2. 全局特征分支 ---------------
        v_global = self.global_branch_v(v_proj.mean(dim=1))  # [B, h_dim]
        q_global = self.global_branch_q(q_proj.mean(dim=1))  # [B, h_dim]
        global_combined = torch.cat([v_global, q_global], dim=-1)  # [B, 2*h_dim]

        # --------------- 3. 融合模块 ---------------
        fused_Wb = self.fusion_Wb(global_combined)
        fused = fused_Wb
        # --------------- 4. 多分支注意力 ---------------
        # v 分支注意力权重
        attn_v = self.attn_branch_v(v_proj).squeeze(-1)  # [B, seq_len_v]，softmax 后权重
        # q 分支注意力权重
        attn_q = self.attn_branch_q(q_proj).squeeze(-1)  # [B, seq_len_q]，softmax 后权重

        # 用注意力加权特征
        v_attn_weighted = torch.sum(v_proj * attn_v.unsqueeze(-1), dim=1)
        q_attn_weighted = torch.sum(q_proj * attn_q.unsqueeze(-1), dim=1)
        attn_combined = torch.cat([v_attn_weighted, q_attn_weighted], dim=-1)

        # --------------- 5. 位置编码（保持原有逻辑） ---------------
        pos_v = self.pos_encoder(v_feat)
        pos_q = self.pos_encoder(q_feat)


        # --------------- 6. 合并所有分支 ---------------
        combined_features = torch.cat([
            fused,
            attn_combined,
        ], dim=-1)

        # --------------- 7. 主输出 + 残差 ---------------
        main_out = self.main_output(combined_features)

        # --------------- 8. 额外注意力图（用于可视化） ---------------
        base_attn = torch.einsum('bkd,bld->bkl', pos_v, pos_q)
        attn_maps = attn_v.unsqueeze(1).unsqueeze(3) * attn_q.unsqueeze(1).unsqueeze(2) * base_attn


        return main_out, attn_maps


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        sin_pos = torch.sin(position * self.div_term)
        cos_pos = torch.cos(position * self.div_term)
        pe = torch.zeros(x.shape, device=x.device)
        pe[:, :, 0::2] = sin_pos
        pe[:, :, 1::2] = cos_pos
        return F.tanh(x + pe)  # 保持位置编码后的激活
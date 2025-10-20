# scm_fusion_xattn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SCMFusionXAttn(nn.Module):
    """
    视觉 Q（每个 RoI 一个 token） × 轨迹 K/V（按帧序列 token）

    输入:
        vis_feat    : (N, Cq)       每个 RoI 的视觉向量（GAP 后）
        traj_tokens : (N, T, Dk)    该 RoI 对应的轨迹帧序列
        traj_mask   : (N, T)        可选，True/1 有效，False/0 为 padding

    输出:
        fused_feat  : (N, Cout)     直接作为 bbox_head 的输入（不做视觉残差）
    """
    def __init__(self,
                 vis_dim: int = 2304,
                 traj_dim: int = 2,        # 每帧轨迹维度（例：2 或 2+1+K）
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 max_seq_len: int = 64,
                 dropout: float = 0.1,
                 output_dim: int = 2304,
                 traj_scale: float = 1.0,  # 注入强度：<1.0 会与原视觉线性插值
                 ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.traj_scale = float(traj_scale)

        # Q/KV 对齐到同一 embed 维
        self.vis_align  = nn.Linear(vis_dim,  embed_dim)
        self.traj_align = nn.Linear(traj_dim, embed_dim)

        # 可学习时间位置编码（给 K/V 用；Q 只有 1 个 token 不需要时间 PE）
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-LN
        self.q_ln  = nn.LayerNorm(embed_dim)
        self.kv_ln = nn.LayerNorm(embed_dim)
        self.out_ln = nn.LayerNorm(embed_dim)

        # 多头注意力投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.out_drop  = nn.Dropout(dropout)

        # FFN
        ffn_dim = 4 * embed_dim
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout),
        )

        # 映射回 bbox_head 需要的通道数
        self.out_align = nn.Linear(embed_dim, output_dim)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self,
                vis_feat: torch.Tensor,          # (N, Cq)
                traj_tokens: torch.Tensor,       # (N, T, Dk)
                traj_mask: Optional[torch.Tensor] = None  # (N, T) True=valid
                ) -> torch.Tensor:
        assert vis_feat.dim() == 2, f"vis_feat 需 (N,C)，got {vis_feat.shape}"
        assert traj_tokens.dim() == 3, f"traj_tokens 需 (N,T,D)，got {traj_tokens.shape}"
        N, T = traj_tokens.size(0), traj_tokens.size(1)

        # 1) 对齐
        q  = self.vis_align(vis_feat).unsqueeze(1)  # (N,1,E)
        kv = self.traj_align(traj_tokens)           # (N,T,E)

        # 2) Pre-LN
        q  = self.q_ln(q)
        kv = self.kv_ln(kv)

        # 3) 时间位置编码（截断到 T）
        maxT = self.pos_embed.size(1)
        if T <= maxT:
            kv = kv + self.pos_embed[:, :T, :]
        else:
            kv[:, :maxT, :] = kv[:, :maxT, :] + self.pos_embed
            # 超出 max_seq_len 的部分不加 PE（可按需改成插值）

        # 4) 多头拆分
        Q = self.q_proj(q).view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)   # (N,h,1,d)
        K = self.k_proj(kv).view(N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (N,h,T,d)
        V = self.v_proj(kv).view(N, T, self.num_heads, self.head_dim).transpose(1, 2)  # (N,h,T,d)

        # 5) 注意力
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (N,h,1,T)
        if traj_mask is not None:
            mask = (~traj_mask.bool()).unsqueeze(1).unsqueeze(2)  # (N,1,1,T)
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, V)                                  # (N,h,1,d)
        y = y.transpose(1, 2).contiguous().view(N, 1, self.embed_dim)  # (N,1,E)

        # 6) 残差（在 embed 空间），再 FFN
        y = q + self.out_drop(self.out_proj(y))                    # (N,1,E)
        y = y + self.ffn(self.out_ln(y))                           # (N,1,E)

        # 7) 输出对齐到 bbox_head 通道；压掉 L=1
        out = self.out_align(y.squeeze(1))                         # (N,Cout)

        # 8) 注入强度（温和起步可设 0.2~0.5；=1.0 就全量用 Cross-Attn 输出）
        if self.traj_scale < 1.0:
            out = (1 - self.traj_scale) * vis_feat + self.traj_scale * out
        return out

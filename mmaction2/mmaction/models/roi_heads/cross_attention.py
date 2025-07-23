import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossAttentionWithLearnablePE(nn.Module):
    def __init__(self,
                 input_dim_q: int,
                 input_dim_k: int,
                 embed_dim: int,
                 num_heads: int,
                 max_seq_len: int = 500,
                 dropout: float = 0.1,
                 ffn_dim: Optional[int] = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 输入对齐层
        self.vis_align = nn.Linear(input_dim_q, embed_dim)
        self.traj_align = nn.Linear(input_dim_k, embed_dim)

        # 可学习绝对位置编码 (核心保留)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-LayerNorm
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_k = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 注意力投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)

        # 前馈网络
        ffn_dim = ffn_dim or embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # 输出对齐层: 将内部 embed_dim 恢复到视觉原始维度
        self.out_align = nn.Linear(embed_dim, input_dim_q)

    def forward(self,
                x: torch.Tensor,       # 视觉特征: [B, L_q, input_dim_q]
                memory: torch.Tensor   # 轨迹特征: [B, L_k, input_dim_k]
                ) -> torch.Tensor:
        B, L_q, _ = x.shape
        L_k = memory.shape[1]

        # 1. 对齐维度
        q = self.vis_align(x)        # [B, L_q, embed_dim]
        k = self.traj_align(memory)  # [B, L_k, embed_dim]
        v = k.clone()                # V与K共享对齐后的特征
        
        # 2. Pre-LN
        q = self.norm1_q(q)
        k = self.norm1_k(k)
        v = self.norm1_k(v)  # V也需要归一化

        # 3. 添加可学习位置编码 (核心保留)
        q = q + self.pos_embed[:, :L_q]
        k = k + self.pos_embed[:, :L_k]
        v = v + self.pos_embed[:, :L_k]  # V也需要位置编码

        # 4. 投影并拆分多头
        Q = self.q_proj(q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 5. 注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 6. 应用注意力到V并合并头
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        # 7. 残差连接 (使用对齐后的 q)
        out = q + attn_output

        # 8. FFN + 残差
        ffn_output = self.ffn(self.norm2(out))
        out = out + ffn_output

        # 9. 输出对齐到视觉原始维度
        out = self.out_align(out)  # [B, L_q, input_dim_q]
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class SCMFusionTransformer(nn.Module):
    """
    SCMFusionTransformer:视觉与轨迹特征融合模块

    参数:
      vis_dim     (int) : 输入视觉特征的维度
      traj_dim    (int) : 输入轨迹特征的维度
      model_dim   (int) : Transformer 内部特征维度（默 256)
      num_layers  (int) : TransformerEncoder 层数（默 2)
      num_heads   (int) : 多头注意力头数（默 8)
      dropout     (float): Dropout 比例（默 0.1)
      output_dim  (int) : 融合后输出特征维度（通常与 Head 通道数相同）
      enhanced    (bool) : 是否在输出后再加 LayerNorm+Dropout
      fusion_type (str) : Token 聚合方式，支持：'mean','cls','concat_mlp','attention','gate'
    """
    def __init__(self,
                 vis_dim: int,
                 traj_dim: int,
                 model_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 output_dim: int = 2304,
                 enhanced: bool = False,
                 fusion_type: str = 'mean'):
        super().__init__()
        # 检查 fusion_type 是否支持
        assert fusion_type in ['mean', 'cls', 'concat_mlp', 'attention', 'gate'], \
            f"Unsupported fusion_type: {fusion_type}"
        self.fusion_type = fusion_type
        self.seq_len = 2  # 序列长度：视觉 + 轨迹 两个 token

        # 1. 增强的特征投影层（Linear + LayerNorm + GELU + Dropout）
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. 可学习的位置编码 & CLS Token
        #    pos_scale：缩放因子，让位置编码的强弱可学习
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        if fusion_type == 'cls':
            # cls_token 形状 (1,1,model_dim)，后面会扩展到批大小
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            # 位置编码长度 = seq_len + 1
            self.pos_embedding = nn.Parameter(torch.zeros(self.seq_len + 1, model_dim))
        else:
            # 位置编码长度 = seq_len
            self.pos_embedding = nn.Parameter(torch.zeros(self.seq_len, model_dim))
        # 使用小标准差正态分布初始化
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # 3. 各种融合策略的专用小模块
        if fusion_type == 'concat_mlp':
            # 拼接后用 MLP 缩回 model_dim
            self.concat_mlp = nn.Sequential(
                nn.Linear(self.seq_len * model_dim, 2 * model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * model_dim, model_dim),
            )
        elif fusion_type == 'attention':
            # 注意力池化：先映到 1 维再 softmax 加权
            self.attn_pool = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim // 2, 1),
            )
        elif fusion_type == 'gate':
            # 门控网络 + 一个残差权重
            self.gate_net = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 1),
            )
            self.res_weight = nn.Parameter(torch.tensor(0.5))

        # 4. Transformer 编码器（多层多头注意力 + 前馈）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN，更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )

        # 5. 输出投影（2x hidden → output_dim）
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, 2 * output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * output_dim, output_dim),
        )

        # 6. 可选的输出增强层
        self.enhanced = enhanced
        if enhanced:
            self.output_ln = nn.LayerNorm(output_dim)
            self.output_dp = nn.Dropout(dropout)

        # 7. 权重初始化：Xavier for Linear，常数 for LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, vis_feat: torch.Tensor, traj_feat: torch.Tensor) -> torch.Tensor:

        """
        vis_feat  : (B, vis_dim)   视觉特征
        traj_feat : (B, traj_dim)  轨迹特征
        返回 fused_feat: (B, output_dim)
        """
        B = vis_feat.size(0)

        # —— 1. 投影 & 激活 —— 
        v = self.vis_proj(vis_feat)     # (B, model_dim)
        t = self.traj_proj(traj_feat)   # (B, model_dim)

        # —— 2. 构建 Transformer 输入序列 —— 
        if self.fusion_type == 'cls':
            # 扩展 cls_token 到 batch 维度
            cls_tok = self.cls_token.expand(B, -1, -1)  # (B,1,model_dim)
            # 拼接成 (B, 3, model_dim)：[cls, vis, traj]
            seq = torch.cat([cls_tok, v.unsqueeze(1), t.unsqueeze(1)], dim=1)
        else:
            # 拼成 (B, 2, model_dim)：[vis, traj]
            seq = torch.stack([v, t], dim=1)

        # —— 3. 添加位置编码 —— 
        # pos_scale * pos_embedding -> (1, seq_len, model_dim) 广播
        seq = seq + self.pos_scale * self.pos_embedding.unsqueeze(0)

        # —— 4. Transformer 编码 —— 
        enc = self.transformer(seq)  # (B, seq_len(+1), model_dim)

        # —— 5. 融合策略 —— 
        if self.fusion_type == 'mean':
            agg = enc.mean(dim=1)                  # 平均
        elif self.fusion_type == 'cls':
            agg = enc[:, 0, :]                    # 取 cls token
        elif self.fusion_type == 'concat_mlp':
            x = enc.reshape(B, -1)                # 展平 (B, seq_len*model_dim)
            agg = self.concat_mlp(x)              # MLP
        elif self.fusion_type == 'attention':
            w = self.attn_pool(enc).squeeze(-1)   # (B, seq_len)
            w = F.softmax(w, dim=1)               # 权重
            agg = (enc * w.unsqueeze(-1)).sum(1)  # 加权和
        else:  # gate
            # 用 Transformer 输出的平均向量算 gate
            mean_tok = enc.mean(dim=1)            # (B, model_dim)
            g = torch.sigmoid(self.gate_net(mean_tok))  # (B,1)
            # 取出 Transformer 输出的 vis/traj 两 token
            v_tok = enc[:, 0, :] if self.fusion_type=='cls' else enc[:,0,:]
            t_tok = enc[:, -1, :]
            # 门控 + 残差
            agg = g * v_tok + (1 - g) * t_tok + self.res_weight * (v_tok + t_tok)

        # —— 6. 输出投影 —— 
        out = self.output_proj(agg)  # (B, output_dim)

        # —— 7. 可选增强 —— 
        if self.enhanced:
            out = self.output_ln(out)
            out = self.output_dp(out)

        return out

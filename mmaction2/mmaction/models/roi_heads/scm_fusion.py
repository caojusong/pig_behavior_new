import torch
import torch.nn as nn
import torch.nn.functional as F

class SCMFusionTransformer(nn.Module):
    """
    视觉 × 轨迹 的 Transformer 融合（两 token：vis、traj，简单聚合版）

    输入:
        vis_feat  : Tensor，形状 (N, vis_dim)
            - 每个 RoI 的视觉向量（例如 RoIAlign + GAP 后的通道向量）
        traj_feat : Tensor，形状 (N, traj_dim)
            - 与该 RoI 一一对应的轨迹向量（例如 TrackMLPEncoder 的输出）

    输出:
        fused_feat: Tensor，形状 (N, output_dim)
            - 融合后的特征，维度通常与 ROI Head 的 in_channels 对齐（如 2304）

    说明:
        1) 先把 vis_feat / traj_feat 通过线性层映射到同一维度 model_dim
        2) 将两者堆叠为一个长度为 2 的序列 [vis, traj]（形状变为 (N, 2, model_dim)）
        3) 加可学习的位置编码（2 个 token 的固定长度编码）
        4) 送入 TransformerEncoder（多头注意力 + FFN）进行双向交互
        5) 对长度为 2 的序列做“简单平均”聚合（也可替换为 CLS/attention 等）
        6) 映射回 output_dim，供 bbox_head 使用
    """

    def __init__(self,
                 vis_dim: int = 2304,
                 traj_dim: int = 2304,
                 model_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 output_dim: int = 2304,
                 fusion_type: str = 'mean',   # ← 新增（即便目前只用 mean）
                 enhanced: bool = False,      # ← 新增（可选输出增强）
                 verbose: bool = False):
        super().__init__()
        self.verbose = bool(verbose)
        self.fusion_type = fusion_type  # ← 新增（即便目前只用 mean）
        self.enhanced = enhanced        # ← 新增（可选输出增强）
        self.seq_len = 2  # 两个 token：vis、traj

        # —— 1) 投影层：把两路特征统一到 model_dim —— #
        # 线性映射 + LN + GELU + Dropout，能让数值更稳定
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

        # —— 2) 可学习位置编码（长度=2）+ 可学习强度 —— #
        self.pos_embedding = nn.Parameter(torch.zeros(self.seq_len, model_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        # —— 3) Transformer 编码器 —— #
        # 即使序列长度只有 2，注意力也能学习“偏向视觉 or 轨迹”的权重关系
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,   # 输入/输出的第一维是 batch（N）
            norm_first=True,    # Pre-LN，更稳定
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )

        # —— 4) 输出映射回 ROI Head 通道数 —— #
        # 用一个小 MLP 提升表达（也可以只用一层 Linear）
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, 2 * output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * output_dim, output_dim),
        )
        # 如果想用 enhanced，就加点简单后处理（可选）
        if self.enhanced:
            self.out_ln = nn.LayerNorm(output_dim)
            self.out_dp = nn.Dropout(dropout)

        self.apply(self._init_weights)
        self._printed = False

        # 初始化一下线性层与 LN
        self.apply(self._init_weights)
        self._printed = False  # 仅首个 forward 打印形状（当 verbose=True 时）

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, vis_feat: torch.Tensor, traj_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vis_feat  (Tensor): (N, vis_dim)，每个 RoI 的视觉特征
            traj_feat (Tensor): (N, traj_dim)，与该 RoI 对齐的轨迹特征
        Returns:
            Tensor: (N, output_dim)，融合后的特征，后续可直接送入 bbox_head
        """
        # —— 基本形状检查 —— #
        if vis_feat.dim() != 2 or traj_feat.dim() != 2:
            raise ValueError(f"vis_feat / traj_feat 必须是二维 (N,D)："
                             f"got {vis_feat.shape} / {traj_feat.shape}")
        if vis_feat.size(0) != traj_feat.size(0):
            raise ValueError(f"N 不一致：vis N={vis_feat.size(0)}, traj N={traj_feat.size(0)}")
        N = vis_feat.size(0)

        # —— 1) 映射到统一维度 —— #
        v = self.vis_proj(vis_feat)    # (N, model_dim)
        t = self.traj_proj(traj_feat)  # (N, model_dim)

        # —— 2) 组序列：[vis, traj] —— #
        seq = torch.stack([v, t], dim=1)  # (N, 2, model_dim)

        # —— 3) 位置编码 —— #
        # 广播到 (N, 2, model_dim)，让两 token 的“角色”可区分
        seq = seq + self.pos_scale * self.pos_embedding.unsqueeze(0)

        # —— 4) Transformer 交互 —— #
        enc = self.transformer(seq)  # (N, 2, model_dim)

        # —— 5) 简单聚合（平均两个 token） —— #
        agg = enc.mean(dim=1)        # (N, model_dim)

        # —— 6) 输出映射 —— #
        out = self.output_proj(agg)  # (N, output_dim)

        # —— 可选：仅第一次打印一下形状，方便核对 —— #
        if self.verbose and not self._printed:
            print("[SCMFusionTransformer] 形状核对：")
            print(f"  vis_feat : {tuple(vis_feat.shape)}  -> proj -> {tuple(v.shape)}")
            print(f"  traj_feat: {tuple(traj_feat.shape)} -> proj -> {tuple(t.shape)}")
            print(f"  seq      : {tuple(seq.shape)}  (N, 2, model_dim)")
            print(f"  enc      : {tuple(enc.shape)}  (N, 2, model_dim)")
            print(f"  out      : {tuple(out.shape)}  (N, output_dim)")
            self._printed = True

        return out

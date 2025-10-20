import torch
import torch.nn as nn
import torch.nn.functional as F
####这个是没有加通道注意力的一版###
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
####下一版####
加了注意力一版
import torch
import torch.nn as nn

class SE1d(nn.Module):
    """Squeeze-Excitation for 1D channel vectors.
       支持 (N,C) 或 (N,T,C)；(N,T,C) 会对 T 做平均得到通道权重。"""
    def __init__(self, channels: int, reduction: int = 16,
                 act=nn.ReLU, gate=nn.Sigmoid):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.act = act()
        self.gate = gate()

    def forward(self, x):
        # x: (N,C) or (N,T,C)
        if x.dim() == 3:
            s = x.mean(dim=1)  # (N,C)
        else:
            s = x              # (N,C)
        w = self.gate(self.fc2(self.act(self.fc1(s))))  # (N,C)
        if x.dim() == 3:
            w = w.unsqueeze(1)  # (N,1,C)
        return x * w


class SCMFusionTransformer(nn.Module):
    """
    两路输入（视觉/轨迹）→ 同维映射 → ① 分支端 SE → 融合（mean/transformer）
                             → ②（可选）融合后 SE → 输出指定维度

    关键改动（稳定训练）：
      1) 投影前 LayerNorm（视觉/轨迹各一）：解决两路特征“尺度不匹配”；
      2) out_proj 全零初始化 + 视觉残差：起步输出≈原视觉分布，不劣于基线；
      3) （可选）轨迹前 3 维做固定缩放，把 dx,dy,dist 先粗归一化再 LN。
    """
    def __init__(self,
                 vis_dim=2304,
                 traj_dim=2304,
                 model_dim=256,
                 num_layers=2,
                 num_heads=8,
                 dropout=0.1,
                 output_dim=2304,
                 enhanced=False,             # 兼容旧字段：可用来打开融合后 SE
                 fusion_type='transformer',  # 'mean' | 'transformer'
                 # —— ① 分支端 SE（推荐默认开）——
                 use_ca_vis=True,
                 use_ca_traj=True,
                 ca_reduction=16,
                 # —— ② 融合后 SE（可选，默认关；若 enhanced=True 则强制打开）——
                 use_ca_post=False,
                 # mean 融合下的轨迹注入比例
                 gate_alpha=0.25):
        super().__init__()
        self.fusion_type  = fusion_type
        self.use_ca_vis   = use_ca_vis
        self.use_ca_traj  = use_ca_traj
        self.use_ca_post  = bool(use_ca_post or enhanced)
        self.gate_alpha   = gate_alpha

        # >>> NEW: 投影前 LayerNorm，统一两路特征的分布（零均值/单位方差）
        self.ln_vis  = nn.LayerNorm(vis_dim)
        self.ln_traj = nn.LayerNorm(traj_dim)

        # 投到统一维度
        self.vis_proj  = nn.Linear(vis_dim,  model_dim)
        self.traj_proj = nn.Linear(traj_dim, model_dim)

        # ① 分支端 SE
        if use_ca_vis:
            self.ca_vis  = SE1d(model_dim, reduction=ca_reduction)
        if use_ca_traj:
            self.ca_traj = SE1d(model_dim, reduction=ca_reduction)

        # 融合模块
        if fusion_type == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads,
                dim_feedforward=4 * model_dim,
                dropout=dropout, activation='gelu',
                batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ② 融合后 SE（可选，残差更稳）
        if self.use_ca_post:
            self.ca_post = SE1d(model_dim, reduction=ca_reduction)

        # 输出到 ROIHead 需要的通道（通常 2304）
        self.out_proj = nn.Linear(model_dim, output_dim)

        # >>> NEW: out_proj 全零初始化，确保初始 fused≈vis_feat（不破坏旧分布）
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, vis_feat, traj_feat):
        # >>> NEW: 保存“原始视觉特征”作残差基线（与 bbox_head 旧分布对齐）
        vis_base = vis_feat

        # >>> NEW (可选)：若轨迹前 3 维是 dx,dy,dist，先做固定缩放以粗归一化
        # dx,dy ∈ [-1.5,1.5]，dist_water ~ [0,2] → 乘 [1/1.5, 1/1.5, 1/2]
        if traj_feat.shape[1] >= 3:
            scale = torch.tensor([1/1.5, 1/1.5, 1/2.0],
                                 device=traj_feat.device, dtype=traj_feat.dtype)
            traj_feat = traj_feat.clone()
            traj_feat[:, :3] = traj_feat[:, :3] * scale

        # 1) 投影前先做 LN，把两路对齐到 ~N(0,1)，减少“尺度不匹配”的优化负担
        v = self.vis_proj(self.ln_vis(vis_feat))   # (N, C)
        t = self.traj_proj(self.ln_traj(traj_feat))# (N, C)

        # 2) ① 分支端 SE（轻权重的通道重标定）
        if self.use_ca_vis:
            v = self.ca_vis(v)
        if self.use_ca_traj:
            t = self.ca_traj(t)

        # 3) 融合（transformer 或稳定的 mean 融合）
        if self.fusion_type == 'transformer':
            seq = torch.stack([v, t], dim=1)   # (N,2,C)
            enc = self.enc(seq)                # (N,2,C)
            agg = enc.mean(dim=1)              # 简单聚合（稳定）
        else:
            alpha = self.gate_alpha
            agg = (1 - alpha) * v + alpha * t  # (N,C)

        # 4) ② 融合后 SE（可选，做成残差）
        if self.use_ca_post:
            agg = agg + self.ca_post(agg)

        # 5) 输出到下游维度 —— 用“视觉残差 + out_proj(增益)”
        #    out_proj 已零初始化 → 初始 fused ≈ vis_base（不惊扰 ROIHead 的统计）
        fused = vis_base + self.out_proj(agg)  # (N, output_dim)
        return fused

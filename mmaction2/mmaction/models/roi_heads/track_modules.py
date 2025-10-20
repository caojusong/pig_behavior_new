import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackMLPEncoder(nn.Module):
    """
    轨迹时间序列编码器：接受 (N, T, D)，逐帧线性映射 → 时间池化 → 定长向量
    ------------------------------------------------
    典型输入:
        D = 2 + 1 + K
          - 0: dx     （平移原点后的）
          - 1: dy
          - 2: water_dist（到饮水器的距离）
          - 3..(2+K): pig_dists（最近 K 个猪的距离，按近到远）

    Args:
        input_dim   (int): 每帧输入维度 D（必须与送入张量最后一维一致，比如 2+1+K）
        hidden_dim  (int): 隐藏维度
        output_dim  (int): 输出维度（与你的 ROI 通道数对齐用，比如 2304）
        dropout_ratio (float)
        select_idx (list[int] or None):
            - None: 用全 D 维（如 2+1+K）
            - 列表: 只取部分维度（例如只取 [0,1] 即 dx,dy；或 [0,1,2] 取 dx,dy+水距）

        verbose (bool): 首次前向打印形状核对
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 1024,
                 output_dim: int = 2304,
                 dropout_ratio: float = 0.5,
                 select_idx=None,
                 verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.select_idx = select_idx

        in_dim = input_dim if select_idx is None else len(select_idx)
        self.input_dim = input_dim
        self.effective_in_dim = in_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._printed = False

    def forward(self, track_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            track_vector: (N, T, D)
        Returns:
            (N, output_dim)
        """
        assert isinstance(track_vector, torch.Tensor), \
            f"track_vector 必须是 torch.Tensor，但得到 {type(track_vector)}"
        assert track_vector.dim() == 3, \
            f"要求形状 (N, T, D)，但得到 {tuple(track_vector.shape)}"
        N, T, D = track_vector.shape

        if self.select_idx is None:
            assert D == self.input_dim, \
                f"D={D} 必须等于 input_dim={self.input_dim}（或传 select_idx 指定子维度）"
            x_in = track_vector
        else:
            assert max(self.select_idx) < D, \
                f"select_idx 越界：D={D}, select_idx={self.select_idx}"
            x_in = track_vector[..., self.select_idx]
            assert x_in.size(-1) == self.effective_in_dim

        # 逐帧线性映射（D -> hidden_dim）
        x = self.linear1(x_in)          # (N, T, hidden_dim)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)

        # 时间池化（平均）
        x = x.mean(dim=1)               # (N, hidden_dim)

        # 输出映射
        x = self.linear2(x)             # (N, output_dim)

        if self.verbose and not self._printed:
            print("[TrackMLPEncoder] 形状核对：")
            print(f"  in : (N,T,D)        = {tuple(track_vector.shape)}  (use {'ALL' if self.select_idx is None else self.select_idx})")
            print(f"  eff_in_dim          = {self.effective_in_dim}")
            print(f"  after fc1+pool      = {(N, self.linear2.in_features)}")
            print(f"  out: (N,output_dim) = {tuple(x.shape)}")
            self._printed = True

        return x

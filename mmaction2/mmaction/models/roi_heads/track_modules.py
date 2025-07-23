import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackMLPEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=1024, output_dim=2304, dropout_ratio=0.5):
        super(TrackMLPEncoder, self).__init__()
        # 第一层线性变换：从 input_dim 到 hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # 第二层线性变换：从 hidden_dim 到 output_dim (2304)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # Dropout 层，训练时按一定概率丢弃，推理时自动关闭:contentReference[oaicite:0]{index=0} 
        self.dropout = nn.Dropout(p=dropout_ratio)
        
    def forward(self, track_vector):
        """
        track_vector: Tensor of shape (N, T, 2) 或者 (T, 2),
                      N 为 track 数量,T 为帧数,2 为偏移向量维度。
        """
        # 如果输入是 (T,2)，扩展为 (1,T,2)
        if track_vector.dim() == 2:
            track_vector = track_vector.unsqueeze(0)
        # 线性变换 + ReLU + Dropout
        x = self.linear1(track_vector)       # (N, T, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x)                  # 训练时丢弃部分神经元，推理时不丢弃:contentReference[oaicite:1]{index=1}
        # 对时间维度做平均池化
        x = x.mean(dim=1)                    # (N, hidden_dim)
        # 输出到固定维度 2304
        x = self.linear2(x)                  # (N, output_dim)
        return x
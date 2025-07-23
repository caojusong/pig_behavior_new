import os
import pandas as pd
import torch
import warnings
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadTrackInfo(BaseTransform):
    """
    轨迹加载 & 编码模块 (3 阶段匹配 + T 帧偏移向量)：
      1. 中心帧 IoU 匹配
      2. ±1 秒窗口内 IoU 最大匹配
      3. 全局平均 IoU 回退
    最后输出 (T,3) 的 track_vector：[dx, dy, dist_water]
    """
    def __init__(self, track_base_path: str, iou_threshold: float = 0.6):
        super().__init__()
        self.track_base_path = track_base_path
        self.iou_threshold = iou_threshold
        self.stats = {'total': 0, 'stage1': 0, 'stage2': 0, 'stage3': 0, 'fail': 0}

        # 饮水器中心点 (归一化) 映射到 [-1,1]
        dx = (0.0056745 - 0.5) * 2  # ≈ -0.98865
        dy = (0.4214915 - 0.5) * 2  # ≈ -0.15702
        self.water_center = torch.tensor([dx, dy], dtype=torch.float32)

    def transform(self, results: dict) -> dict:
        video_id = results['video_id']
        fps = results['fps']
        frame_inds = results['frame_inds'].tolist()
        gt_box = results['gt_bboxes'][0].tolist()
        center_idx = len(frame_inds) // 2
        center_frm = frame_inds[center_idx]
        self.stats['total'] += 1

        # 1) 加载跟踪文件
        txt_path = os.path.join(self.track_base_path, f"{video_id}.txt")
        if not os.path.isfile(txt_path):
            warnings.warn(f"[LoadTrackInfo] Missing track file: {txt_path}")
            return self._fallback_to_zero(results, len(frame_inds))

        df = pd.read_csv(txt_path, header=None,
                         names=['frame','tid','x1','y1','w','h','score','d1','d2','d3'])
        first_frame = int(df.iloc[0]['frame']) if len(df) > 0 else 0
        offset = -1 if first_frame == 1 else 0
        df['fid'] = df['frame'] + offset

        # 2) 三阶段 IoU 匹配确定 best_tid
        best_tid = None
        # Stage1: 中心帧
        df_c = df[df['fid'] == center_frm]
        best_iou = 0.0
        for _, r in df_c.iterrows():
            box = [r.x1, r.y1, r.x1 + r.w, r.y1 + r.h]
            iou = self._iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_tid = iou, int(r.tid)
        if best_iou >= self.iou_threshold:
            self.stats['stage1'] += 1
        else:
            # Stage2: ±1 秒窗口
            window_start = center_frm - fps
            window_end   = center_frm + fps
            df_w = df[(df['fid'] >= window_start) & (df['fid'] <= window_end)]
            best_iou2 = 0.0
            for _, r in df_w.iterrows():
                box = [r.x1, r.y1, r.x1 + r.w, r.y1 + r.h]
                iou = self._iou(box, gt_box)
                if iou > best_iou2:
                    best_iou2, best_tid = iou, int(r.tid)
            if best_iou2 >= self.iou_threshold:
                self.stats['stage2'] += 1
            else:
                # Stage3: 全局平均
                tid_stats = {}
                for fid in frame_inds:
                    df_f = df[df['fid'] == fid]
                    for _, r in df_f.iterrows():
                        box = [r.x1, r.y1, r.x1 + r.w, r.y1 + r.h]
                        iou = self._iou(box, gt_box)
                        tid = int(r.tid)
                        if tid not in tid_stats:
                            tid_stats[tid] = {'sum': 0.0, 'cnt': 0}
                        tid_stats[tid]['sum'] += iou
                        tid_stats[tid]['cnt'] += 1
                best_avg, best_tid3 = 0.0, None
                for tid, st in tid_stats.items():
                    avg_iou = st['sum'] / st['cnt']
                    if avg_iou > best_avg:
                        best_avg, best_tid3 = avg_iou, tid
                if best_avg >= self.iou_threshold:
                    best_tid = best_tid3
                    self.stats['stage3'] += 1
                else:
                    self.stats['fail'] += 1
                    best_tid = None

        # 3) 计算 dx, dy
        H, W = results['img_shape']
        traj = []
        last = [0.0, 0.0]
        for fid in frame_inds:
            df_f = df[df['fid'] == fid]
            if best_tid is not None and (df_f['tid'] == best_tid).any():
                r = df_f[df_f['tid'] == best_tid].iloc[0]
                cx = (r.x1 + (r.x1 + r.w)) * 0.5
                cy = (r.y1 + (r.y1 + r.h)) * 0.5
                dx_ = (cx - W/2) / (W/2)
                dy_ = (cy - H/2) / (H/2)
                last = [dx_, dy_]
            traj.append(last)
        traj_tensor = torch.tensor(traj, dtype=torch.float32)

        # 4) 计算到饮水器距离 dist_water
        device = traj_tensor.device
        water_center = self.water_center.to(device)
        dist_water = torch.norm(traj_tensor - water_center, dim=1, keepdim=True)  # (T,1)

        # 5) 拼接 → (T,3)
        fused = torch.cat([traj_tensor, dist_water], dim=1)
        results['track_vector'] = fused

        return results

    @staticmethod
    def _iou(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 1e-6 else 0.0

    def _fallback_to_zero(self, results, num_frames):
        """匹配失败时返回全零 (T,3)。"""
        results['track_vector'] = torch.zeros((num_frames, 3), dtype=torch.float32)
        return results

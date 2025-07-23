import os
import pandas as pd
import torch
import warnings
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadTrackInfo(BaseTransform):
    """
    轨迹加载 & 编码模块(3 阶段匹配 + 32 帧偏移向量）：
      1. 中心帧 IoU 匹配
      2. ±1 秒窗口内 IoU 最大匹配
      3. 全局平均 IoU 回退
    最后输出 (32,2) 的 track_vector。
    """
    def __init__(self, track_base_path: str, iou_threshold: float = 0.3):
        super().__init__()
        self.track_base_path = track_base_path
        self.iou_threshold = iou_threshold

    def transform(self, results: dict) -> dict:
        video_id = results['video_id']
        fps = results['fps']

        # 1) SampleAVAFrames 采样结果
        frame_inds = results['frame_inds'].tolist()
        print(f"[{video_id}] Sampled frames: {frame_inds}")

        # 2) RawFrameDecode 解码后的 GT bboxes（像素坐标）
        gt_bboxes_px = results['gt_bboxes']  # (N,4)
        gt_box = gt_bboxes_px[0].tolist()
        print(f"[{video_id}] Decoded GT (px): {gt_bboxes_px}")

        # 3) 中心帧
        center_idx = len(frame_inds)//2
        center_frm = frame_inds[center_idx]
        print(f"[{video_id}] Center frame: {center_frm}")

        # 4) 加载 ByteTrack 结果
        txt_path = os.path.join(self.track_base_path, f"{video_id}.txt")
        if not os.path.isfile(txt_path):
            warnings.warn(f"[LoadTrackInfo] Missing track file: {txt_path}")
            results['track_vector'] = torch.zeros((len(frame_inds), 2))
            return results

        df = pd.read_csv(txt_path, header=None,
                         names=['frame','tid','x1','y1','w','h','score','d1','d2','d3'])
        offset = -1 if int(df.iloc[0]['frame']) == 1 else 0
        df['fid'] = df['frame'] + offset

        # ---- 阶段1：中心帧 IoU 匹配 ----
        df_c = df[df['fid']==center_frm]
        best_tid, best_iou = None, 0.0
        for _, r in df_c.iterrows():
            box = [r.x1, r.y1, r.x1+r.w, r.y1+r.h]
            iou = self._iou(box, gt_box)
            print(f"  [Center] T{int(r.tid)} IoU={iou:.3f}")
            if iou > best_iou:
                best_iou, best_tid = iou, int(r.tid)

        if best_iou >= self.iou_threshold:
            print(f"[{video_id}] Stage1 matched T{best_tid} (IoU={best_iou:.3f})")
        else:
            print(f"[{video_id}] Stage1 fail (best IoU={best_iou:.3f}), go Stage2")

            # ---- 阶段2：±fps 窗口最大 IoU ----
            window = set(range(center_frm-fps, center_frm+fps+1))
            cand = df[df['fid'].isin(window)]
            best_tid2, best_iou2 = None, 0.0
            for _, r in cand.iterrows():
                box = [r.x1, r.y1, r.x1+r.w, r.y1+r.h]
                iou = self._iou(box, gt_box)
                if iou > best_iou2:
                    best_iou2, best_tid2 = iou, int(r.tid)
            if best_iou2 >= self.iou_threshold:
                best_tid, best_iou = best_tid2, best_iou2
                print(f"[{video_id}] Stage2 matched T{best_tid} (IoU={best_iou:.3f})")
            else:
                print(f"[{video_id}] Stage2 fail (best IoU={best_iou2:.3f}), go Stage3")

                # ---- 阶段3：全局平均 IoU 回退 ----
                stats = {}
                for fid in frame_inds:
                    sub = df[df['fid']==fid]
                    for _, r in sub.iterrows():
                        box = [r.x1, r.y1, r.x1+r.w, r.y1+r.h]
                        iou = self._iou(box, gt_box)
                        rec = stats.setdefault(int(r.tid), {'sum':0.0,'cnt':0})
                        rec['sum'] += iou
                        rec['cnt'] += 1
                if stats:
                    # 平均 IoU
                    best, best_avg = None, 0.0
                    for tid, rec in stats.items():
                        avg = rec['sum']/rec['cnt']
                        if avg > best_avg:
                            best, best_avg = tid, avg
                    if best_avg >= self.iou_threshold:
                        best_tid, best_iou = best, best_avg
                        print(f"[{video_id}] Stage3 matched T{best_tid} (avgIoU={best_iou:.3f})")
                    else:
                        print(f"[{video_id}] Stage3 fail (avgIoU={best_avg:.3f}), no match")
                        best_tid = None
                else:
                    print(f"[{video_id}] Stage3 no candidates")
                    best_tid = None

        # 构造 32 帧轨迹向量
        H, W = results['img_shape']
        traj = []
        last = [0.0, 0.0]
        for fid in frame_inds:
            sub = df[df['fid']==fid]
            if best_tid is not None and (sub['tid']==best_tid).any():
                r = sub[sub['tid']==best_tid].iloc[0]
                cx = (r.x1 + (r.x1+r.w)) * 0.5
                cy = (r.y1 + (r.y1+r.h)) * 0.5
                dx = (cx - W/2)/(W/2)
                dy = (cy - H/2)/(H/2)
                last = [float(dx), float(dy)]
            traj.append(last)

        results['track_vector'] = torch.tensor(traj, dtype=torch.float32)
        return results

    @staticmethod
    def _iou(b1, b2):
        ix1, iy1 = max(b1[0],b2[0]), max(b1[1],b2[1])
        ix2, iy2 = min(b1[2],b2[2]), min(b1[3],b2[3])
        inter = max(0, ix2-ix1)*max(0, iy2-iy1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

import os
import numpy as np
import warnings
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS
import time  
@TRANSFORMS.register_module()
class LoadTrackInfo(BaseTransform):
    """
    改进版轨迹加载Transform:
    1. 所有帧使用全局帧号匹配轨迹文件
    2. 增强轨迹匹配鲁棒性（多帧验证+轨迹连续性检查）
    3. 更安全的视频ID解析
    """

    def __init__(self, track_base_path: str, iou_threshold=0.5, continuity_check=3):
        super().__init__()
        self.track_base_path = track_base_path
        self.iou_threshold = iou_threshold
        self.continuity_check = continuity_check  # 轨迹连续性检查帧数

    def transform(self, results: dict) -> dict:
        # 关键参数提取
        video_id = results['video_id']
        fps = results['fps']
        ts = results['timestamp']
        ts_start = results.get('timestamp_start', 0)
        start_index = results.get('start_index', 0)
        
        # 1. 计算全局帧号体系 ----------------------------------------------
        #seg_start = self._parse_seg_start(video_id)  # 安全解析起始时间
        #global_offset = int(seg_start * fps)  # 全局帧偏移量
        global_offset  = 0 
        # 中心帧全局计算
        center_rel = int((ts - ts_start) * fps) + start_index
        center_global = global_offset + center_rel
        
        # 采样帧全局转换
        frame_inds_global = [global_offset + idx for idx in results['frame_inds']]
        # === 添加调试输出 ===
        #print(f"视频ID: {video_id}")
        #print(f"中心帧: {center_global}")
        #print(f"采样帧数: {len(frame_inds_global)}")
        #print(f"采样帧范围: {min(frame_inds_global)}-{max(frame_inds_global)}")
        #print(f"时间戳: {ts} (当前), {ts_start} (起始)")
            # ===================
        # 2. 准备GT框 ----------------------------------------------------
        img_shape = results.get('img_shape', results.get('original_shape'))
        H, W = img_shape[:2]
        gt_norm = results['gt_bboxes'][0]
        gt_box = [
            float(gt_norm[0]) * W,
            float(gt_norm[1]) * H,
            float(gt_norm[2]) * W,
            float(gt_norm[3]) * H
        ]
        
        # 3. 加载轨迹数据 -------------------------------------------------
        track_file = os.path.join(self.track_base_path, f"{video_id}.txt")
        if not os.path.isfile(track_file):
            warnings.warn(f"轨迹文件缺失: {track_file}")
            return self._fallback_track(results)
        
        # 只读取需要的帧数据
        needed_frames = set([center_global] + frame_inds_global)
        detections = self._load_detections(track_file, needed_frames)
        
        # 4. 鲁棒的目标匹配策略 --------------------------------------------
        candidate_tid = self._find_best_track(
            detections=detections,
            target_frame=center_global,
            gt_box=gt_box,
            fps=fps
        )
        ### <<< DEBUG 打印中心帧匹配结果 >>>
        print(f"[{time.strftime('%H:%M:%S')}][TrackMatch] {video_id}  "
            f"center={center_global}  tid={candidate_tid}  "
            f"{'OK' if candidate_tid is not None else 'FAIL'}",
            flush=True)
        
        # 5. 提取轨迹向量 -------------------------------------------------
        if candidate_tid is None:
            return self._fallback_track(results)
            
        track_vector = self._extract_trajectory(
            detections=detections,
            frame_inds=frame_inds_global,
            track_id=candidate_tid,
            img_size=(W, H)
        )
        
        # -------- 把 numpy → torch.Tensor，以便后续 .to(device) ----------
        import torch
        results['track_vector'] = torch.as_tensor(
            track_vector, dtype=torch.float32)
        return results

    # --------------- 工具函数 ---------------

    def _load_detections(self, track_file, needed_frames):
        """高效加载需要的检测数据"""
        detections = {}
        with open(track_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_idx = int(parts[0])
                if frame_idx not in needed_frames:
                    continue
                    
                tid = int(parts[1])
                x1, y1, w, h = map(float, parts[2:6])
                box = [x1, y1, x1+w, y1+h]
                
                if frame_idx not in detections:
                    detections[frame_idx] = {}
                detections[frame_idx][tid] = box
        return detections

    def _calculate_iou(self, box1, box2):
        """计算两个框的IoU"""
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        
        return inter_area / (area1 + area2 - inter_area) if (area1+area2-inter_area) > 0 else 0

    def _find_best_track(self, detections, target_frame, gt_box, fps):
        """鲁棒的目标匹配策略"""
        # 策略1：中心帧优先匹配
        if target_frame in detections:
            for tid, box in detections[target_frame].items():
                iou = self._calculate_iou(box, gt_box)
                if iou >= self.iou_threshold:
                    return tid
        
        # 策略2：邻近帧搜索（前后1秒）
        search_frames = sorted(detections.keys())
        nearby_frames = [
            f for f in search_frames
            if abs(f - target_frame) <= fps  # 1秒内帧
        ]
        
        best_tid, best_iou = None, 0
        for frame in nearby_frames:
            for tid, box in detections[frame].items():
                iou = self._calculate_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
                    
        if best_iou >= self.iou_threshold * 0.7:  # 降低阈值
            return best_tid
            
        # 策略3：最后手段 - 选择最近帧最大目标
        if nearby_frames:
            last_frame = max(nearby_frames)
            largest_tid = max(
                detections[last_frame].items(),
                key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1])
            )[0]
            print(f"    → choose tid={largest_tid}  (largest box)", flush=True)
            return largest_tid
        
        print("    → no valid track found", flush=True)    
        return None

    def _extract_trajectory(self, detections, frame_inds, track_id, img_size):
        """提取轨迹向量（带连续性检查）"""
        W, H = img_size
        trajectory = []
        last_valid = [0, 0]  # 最后有效偏移
        
        for i, fid in enumerate(frame_inds):
            # 当前帧有检测
            if fid in detections and track_id in detections[fid]:
                box = detections[fid][track_id]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                dx = np.clip((cx - W/2) / (W/2), -1.0, 1.0)
                dy = np.clip((cy - H/2) / (H/2), -1.0, 1.0)
                last_valid = [dx, dy]
                trajectory.append([dx, dy])
                continue
                
            # 当前帧缺失 - 检查轨迹连续性
            if self._check_track_continuity(detections, frame_inds, i, track_id):
                trajectory.append(last_valid)  # 使用上一有效值
            else:
                trajectory.append([0, 0])  # 轨迹中断归零
                ### <<< DEBUG 打印轨迹简要 >>>
        head = trajectory[:2] if len(trajectory) >= 2 else trajectory
        tail = trajectory[-2:] if len(trajectory) >= 2 else trajectory
        print(f"    track_vec shape={len(trajectory)}x2  head={head}  tail={tail}",
            flush=True)
        return np.array(trajectory, dtype=np.float32)

    def _check_track_continuity(self, detections, frame_inds, current_idx, track_id):
        """检查轨迹是否真正中断"""
        # 检查后续帧是否存在该轨迹
        future_frames = frame_inds[current_idx+1 : current_idx+self.continuity_check+1]
        for fid in future_frames:
            if fid in detections and track_id in detections[fid]:
                return True
                
        # 检查前面帧是否存在该轨迹
        past_frames = frame_inds[max(0, current_idx-self.continuity_check) : current_idx]
        for fid in past_frames:
            if fid in detections and track_id in detections[fid]:
                return True
                
        return False

    def _fallback_track(self, results):
        """后备方案：返回零向量"""
        num_frames = len(results['frame_inds'])
        results['track_vector'] = np.zeros((num_frames, 2), dtype=np.float32)
        return results
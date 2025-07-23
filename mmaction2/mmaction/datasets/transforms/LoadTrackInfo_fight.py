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
    def __init__(self, track_base_path: str, iou_threshold: float = 0.6):
        super().__init__()
        self.track_base_path = track_base_path
        self.iou_threshold = iou_threshold
        self.stats = {'total': 0, 'stage1': 0, 'stage2': 0, 'stage3': 0, 'fail': 0}
        # 饮水器中心点：从归一化 [0,1] 映射到图像中心为 (0,0) 的坐标系 → [-1, 1]
        # 归一化中心点为 (0.0056745, 0.4214915)
        dx = (0.0056745 - 0.5) * 2  # ≈ -0.988651
        dy = (0.4214915 - 0.5) * 2  # ≈ -0.157017
        self.water_center = torch.tensor([dx, dy], dtype=torch.float32)  # (-0.988651, -0.157017)
#self.dist_threshold = 0.25  # ✅ 新增距离阈值
# —— 打斗检测：对每帧，取与主track最近的 K 头猪距离 —— 
#self.fight_k = 5  # K=5，你也可以调成 3、10，越大维度越高
    def transform(self, results: dict) -> dict:
        video_id = results['video_id']
        fps = results['fps']
        self.stats['total'] += 1

        # 1) 获取关键参数
        frame_inds = results['frame_inds'].tolist()
        gt_bboxes_px = results['gt_bboxes']  # (N,4)
        gt_box = gt_bboxes_px[0].tolist()
        center_idx = len(frame_inds) // 2
        center_frm = frame_inds[center_idx]
        
        print(f"[{video_id}] Sampled frames: {frame_inds}")
        print(f"[{video_id}] Decoded GT (px): {gt_bboxes_px.tolist()}")
        print(f"[{video_id}] Center frame: {center_frm} (Index {center_idx})")

        # 2) 加载 ByteTrack 结果
        txt_path = os.path.join(self.track_base_path, f"{video_id}.txt")
        if not os.path.isfile(txt_path):
            warnings.warn(f"[LoadTrackInfo] Missing track file: {txt_path}")
            return self._fallback_to_zero(results, len(frame_inds))

        try:
            # 加载并处理跟踪数据
            df = pd.read_csv(txt_path, header=None,
                             names=['frame','tid','x1','y1','w','h','score','d1','d2','d3'])
            
            # 确定帧索引偏移 (0-based vs 1-based)
            first_frame = int(df.iloc[0]['frame']) if len(df) > 0 else 0
            offset = -1 if first_frame == 1 else 0
            df['fid'] = df['frame'] + offset
            
            print(f"[{video_id}] Loaded {len(df)} detections (first frame={first_frame}, offset={offset})")
        except Exception as e:
            warnings.warn(f"[LoadTrackInfo] Failed to process track file: {str(e)}")
            return self._fallback_to_zero(results, len(frame_inds))

        # 3) 三级匹配策略
        best_tid = None
        
        # 阶段1: 中心帧匹配
        df_c = df[df['fid'] == center_frm]
        best_iou = 0.0
        for _, r in df_c.iterrows():
            box = [r.x1, r.y1, r.x1 + r.w, r.y1 + r.h]
            iou = self._iou(box, gt_box)
            print(f"  [Stage1] Frame {r.fid} T{int(r.tid)}: IoU={iou:.4f}")
            if iou > best_iou:
                best_iou, best_tid = iou, int(r.tid)
        
        if best_iou >= self.iou_threshold:
            self.stats['stage1'] += 1
            print(f"[{video_id}] Stage1 matched T{best_tid} (IoU={best_iou:.4f})")
        else:
            print(f"[{video_id}] Stage1 fail (best IoU={best_iou:.4f}), go Stage2")
            
            # 阶段2: ±1 秒窗口匹配
            window_start = center_frm - fps
            window_end = center_frm + fps + 1
            df_window = df[(df['fid'] >= window_start) & (df['fid'] <= window_end)]
            
            best_iou2 = 0.0
            for _, r in df_window.iterrows():
                box = [r.x1, r.y1, r.x1 + r.w, r.y1 + r.h]
                iou = self._iou(box, gt_box)
                if iou > best_iou2:
                    best_iou2, best_tid = iou, int(r.tid)
            
            if best_iou2 >= self.iou_threshold:
                self.stats['stage2'] += 1
                print(f"[{video_id}] Stage2 matched T{best_tid} (IoU={best_iou2:.4f})")
            else:
                print(f"[{video_id}] Stage2 fail (best IoU={best_iou2:.4f}), go Stage3")
                
                # 阶段3: 全局平均 IoU 回退
                tid_stats = {}
                for fid in frame_inds:
                    df_frame = df[df['fid'] == fid]
                    for _, r in df_frame.iterrows():
                        box = [r.x1, r.y1, r.x1 + r.w, r.y1 + r.h]
                        iou = self._iou(box, gt_box)
                        tid = int(r.tid)
                        if tid not in tid_stats:
                            tid_stats[tid] = {'sum': 0.0, 'cnt': 0}
                        tid_stats[tid]['sum'] += iou
                        tid_stats[tid]['cnt'] += 1
                
                if tid_stats:
                    best_avg, best_tid3 = 0.0, None
                    for tid, stats in tid_stats.items():
                        avg_iou = stats['sum'] / stats['cnt']
                        print(f"  [Stage3] T{tid}: avgIoU={avg_iou:.4f}")
                        if avg_iou > best_avg:
                            best_avg, best_tid3 = avg_iou, tid
                    
                    if best_avg >= self.iou_threshold:
                        self.stats['stage3'] += 1
                        best_tid = best_tid3
                        print(f"[{video_id}] Stage3 matched T{best_tid} (avgIoU={best_avg:.4f})")
                    else:
                        self.stats['fail'] += 1
                        print(f"[{video_id}] Stage3 fail (best avgIoU={best_avg:.4f})")
                else:
                    self.stats['fail'] += 1
                    print(f"[{video_id}] Stage3 no candidates")

        # 4) 构造轨迹向量
        H, W = results['img_shape']
        traj = []
        last = [0.0, 0.0]  # 初始位置
        missing_count = 0
        
        for fid in frame_inds:
            df_f = df[df['fid'] == fid]
            if best_tid is not None and (df_f['tid'] == best_tid).any():
                r = df_f[df_f['tid'] == best_tid].iloc[0]
                cx = (r.x1 + (r.x1 + r.w)) * 0.5
                cy = (r.y1 + (r.y1 + r.h)) * 0.5
                dx = (cx - W/2) / (W/2)  # 归一化到 [-1, 1]
                dy = (cy - H/2) / (H/2)
                last = [float(dx), float(dy)]
            else:
                # 使用上一帧位置
                missing_count += 1
            
            traj.append(last)
        
        # 记录匹配统计
        if best_tid is not None:
            coverage = 1.0 - (missing_count / len(frame_inds))
            print(f"[{video_id}] Trajectory coverage: {coverage:.1%} ({len(frame_inds)-missing_count}/{len(frame_inds)} frames)")
        else:
            print(f"[{video_id}] Using zero vector")
        
        #results['track_vector'] = torch.tensor(traj, dtype=torch.float32)
        traj_tensor = torch.tensor(traj, dtype=torch.float32)  # (32, 2)

# 1. 将饮水器坐标放到相同 device
#water_center = self.water_center.to(traj_tensor.device)  # (2,)

 # 2. 计算每帧到饮水器中心点的欧氏距离（单位: 相对中心）
#dist_vec = torch.norm(traj_tensor - water_center, dim=1, keepdim=True)  # → (32, 1)

# 3. 拼接为 (32, 3)：[dx, dy, dist]
#fused_traj = torch.cat([traj_tensor, dist_vec], dim=1)  # → (32, 3)
# ✅ 3. 是否靠近饮水器（二值）
#is_near = (dist_vec < self.dist_threshold).float()  # (32, 1)，靠近饮水器记为1
#fused_traj = torch.cat([traj_tensor, dist_vec, is_near], dim=1)  # → (32, 4)
# —— A. 计算基础轨迹 & 饮水特征 —— 
#traj_tensor = torch.tensor(traj, dtype=torch.float32)  # (T,2)
# dist_vec = torch.norm(traj_tensor - self.water_center.to(traj_tensor.device),dim=1, keepdim=True)            # (T,1)
#is_near = (dist_vec < self.dist_threshold).float()      # (T,1)

# —— B. 计算“打斗”向量，每帧与其它猪最小 K 距 —— 
        #T = traj_tensor.shape[0]
        #fight_list = []
        #for idx, fid in enumerate(frame_inds):
            #df_f = df[df['fid'] == fid]
            #dx0, dy0 = traj_tensor[idx].tolist()
            #dists = []
            #for _, r in df_f.iterrows():

                #if int(r.tid) == best_tid:
                 #   continue
                #cx, cy = r.x1 + r.w/2, r.y1 + r.h/2
                #dx1 = (cx - W/2) / (W/2)
                #dy1 = (cy - H/2) / (H/2)
                #dists.append(((dx0-dx1)**2 + (dy0-dy1)**2)**0.5)
            #dists = sorted(dists)
            #topk = dists[:self.fight_k] + [0.0] * max(0, self.fight_k - len(dists))
            #fight_list.append(topk)
        #fight_vec = torch.tensor(fight_list, dtype=torch.float32,
        #                            device=traj_tensor.device)  # (T, K) """
  
        # —— C. 最终拼接：dx,dy,dist,is_near + fight_vec —— 
        #fused_traj = torch.cat([traj_tensor, dist_vec, is_near, fight_vec], dim=1)
        #results['track_vector'] = fused_traj
        #print(f"[{video_id}] Final track_vector shape: {fused_traj.shape}")
  
        # —— 计算 A. 饮水器距离 & B. 最近猪距离 —— 
        device = traj_tensor.device
        # A. 计算到饮水器的距离 (T,1)
        water_center = self.water_center.to(device)  # (2,)
        dist_water = torch.norm(traj_tensor - water_center, dim=1, keepdim=True)

        # B. 计算每帧与其它猪的最近距离 (T,1)
        dist_pig2pig_list = []
        for idx, fid in enumerate(frame_inds):
            df_f = df[df['fid'] == fid]
            dx0, dy0 = traj_tensor[idx].tolist()
            # 收集所有非主 track 的距离
            dists = []
            for _, r in df_f.iterrows():
                if int(r.tid) == best_tid:
                    continue
                cx = (r.x1 + r.x1 + r.w) * 0.5
                cy = (r.y1 + r.y1 + r.h) * 0.5
                dx1 = (cx - W/2) / (W/2)
                dy1 = (cy - H/2) / (H/2)
                dists.append(((dx0 - dx1)**2 + (dy0 - dy1)**2)**0.5)
            min_d = min(dists) if dists else 0.0
            dist_pig2pig_list.append(min_d)
        dist_pig2pig = torch.tensor(dist_pig2pig_list,
                                    dtype=torch.float32,
                                    device=device).unsqueeze(1)

        # C. 最终拼接成 4 维：(dx, dy, dist_pig2pig, dist_water)
        fused_traj = torch.cat([traj_tensor, dist_pig2pig, dist_water], dim=1)
        # 4. 存入结果
        results['track_vector'] = fused_traj
        print(f"[{video_id}] Final track_vector shape: {fused_traj.shape}")
        print(f"[{video_id}] Example track_vector[center]: {fused_traj[len(fused_traj)//2].tolist()}")

        # 定期报告全局统计
        if self.stats['total'] % 100 == 0:
            total = self.stats['total']
            print(f"\n[Global Stats] Videos: {total}")
            print(f"  Stage1 success: {self.stats['stage1']} ({self.stats['stage1']/total:.1%})")
            print(f"  Stage2 success: {self.stats['stage2']} ({self.stats['stage2']/total:.1%})")
            print(f"  Stage3 success: {self.stats['stage3']} ({self.stats['stage3']/total:.1%})")
            print(f"  Failures: {self.stats['fail']} ({self.stats['fail']/total:.1%})\n")
        
        return results

    @staticmethod
    def _iou(b1, b2):
        """计算两个边界框的交并比"""
        # 计算交集区域
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        
        # 检查是否有交集
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        
        # 计算交集面积
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        
        # 计算各自面积
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        
        # 计算并集面积
        union_area = area1 + area2 - inter_area
        
        # 避免除零错误
        if union_area < 1e-6:
            return 0.0
        
        return inter_area / union_area

    def _fallback_to_zero(self, results, num_frames):
        """匹配失败时返回零向量"""
        #results['track_vector'] = torch.zeros((num_frames, 2), dtype=torch.float32)
        results['track_vector'] = torch.zeros((num_frames, 4), dtype=torch.float32)
        return results
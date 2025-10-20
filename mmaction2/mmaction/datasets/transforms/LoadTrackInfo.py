# -*- coding: utf-8 -*-
"""
多 ROI 逐帧重匹配版 LoadTrackInfo（加入“可平移原点”+ 饮水器距离 + 猪-猪距离）

核心流程（不变）：
  - 同一关键秒若有 M 个 GT(例如 9 头猪），为每个 GT 独立生成一条轨迹 (T,2)
  - 与 AVA/rawframes、TXT 的“1 基帧号”严格对齐（直接用 SampleAVAFrames 输出的 frame_inds/key_fid/center_fid)
  - 起点在关键帧 key_fid 上用 GT 配(IoU 最大），再“一步投射”到本段中心 center_fid, 最后双向逐帧跟随
  - 不依赖 tid, tid 仅用于调试；最终得到的轨迹是 (dx,dy) ∈ [-1,1]

本版新增：
  - 可“平移原点”：将原点从图像中心沿“中心→饮水器”的方向平移 α（origin_alpha）倍
  - 时序特征：
     A) dist_water(T,1)：每帧 ROI 到“饮水器”的欧氏距离（平移前后等价）
     B) pig_dists(T,K)：每帧 ROI 与其它猪的欧氏距离（按最近排序取 K，不足 K 填 0.0）
  - 最终输出：
     results['track_vector'] = (M,T, 2 + 1 + K) = [dx, dy, dist_water, pig_1..K]
     以及 water_dist / pig_dists / 调试信息等
"""

import os
import warnings
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadTrackInfo(BaseTransform):
    """
    参数：
        track_base_path (str):
            跟踪 TXT 目录，每个视频一个文件：{video_id}.txt
            行格式至少包含：frame, tid, x1, y1, w, h, ...

        iou_threshold (float):
            起点匹配（在 key_fid 上 GT→候选）的最小 IoU

        follow_iou_ratio (float|None):
            逐帧跟随“上一帧框”的最小 IoU（低于此值视为 miss，沿用上一帧）
            None 时默认 0.5 * iou_threshold

        roi_index_key (str|None):
            若提供该键（例如 'roi_index'），仅为该 GT 生成一条轨迹；否则为关键秒上的全部 GT 逐一生成

        verbose (bool), log_every_frame (bool), track_frame_offset (int), print_prefix (str):
            调试打印相关，可沿用你原本的习惯

        # ==== 饮水器 & 最近邻 ====
        max_neighbors (int):
            每帧保留的最近邻猪数量 K（不足 K 用 0.0 padding）

        water_center_norm (Tuple[float, float]):
            饮水器在图像归一化坐标(0~1)里的位置（如 (0.0056745,0.4214915)）

        origin_alpha (float in [0,1]):
            原点平移比例 α：
              - 0.0：原点=图像中心（旧版完全一致）
              - 0.5：原点=图像中心与饮水器中心的“中点”
              - 1.0：原点=饮水器中心
            新坐标 = 旧坐标 - α * W，其中 W 为饮水器在 [-1,1] 坐标下的位置向量
    """

    def __init__(self,
                 track_base_path: str,
                 iou_threshold: float = 0.3,
                 follow_iou_ratio: Optional[float] = None,
                 roi_index_key: Optional[str] = 'roi_index',
                 verbose: bool = True,
                 log_every_frame: bool = False,
                 track_frame_offset: int = 0,
                 print_prefix: str = "[TRK]",
                 # 新增
                 max_neighbors: int = 0,
                 water_center_norm: Tuple[float, float] = (0.0056745, 0.4214915),
                 origin_alpha: float = 0.5):
        super().__init__()
        self.track_base_path = track_base_path
        self.iou_threshold = float(iou_threshold)
        self.follow_iou_ratio = (float(follow_iou_ratio)
                                 if follow_iou_ratio is not None
                                 else self.iou_threshold * 0.5)
        self.roi_index_key = roi_index_key
        self.verbose = bool(verbose)
        self.log_every_frame = bool(log_every_frame)
        self.offset = int(track_frame_offset)
        self.print_prefix = str(print_prefix)

        # 最近邻 & 饮水器
        self.max_neighbors = int(max_neighbors)
        wx, wy = float(water_center_norm[0]), float(water_center_norm[1])  # [0,1]
        # 饮水器在 [-1,1] 坐标（图像中心为原点）
        self.water_dx = (wx - 0.5) * 2.0
        self.water_dy = (wy - 0.5) * 2.0

        # 原点平移比例 α ∈ [0,1]：在 [-1,1] 坐标中，实际要平移的量
        self.origin_alpha = float(np.clip(origin_alpha, 0.0, 1.0))
        self.shift_dx = self.origin_alpha * self.water_dx  # 要减去的偏移（应用到所有 dx）
        self.shift_dy = self.origin_alpha * self.water_dy  # 要减去的偏移（应用到所有 dy）

    # ====================== 主流程 ====================== #
    def transform(self, results: dict) -> dict:
        """
        输入（来自上游 SampleAVAFrames + AVADataset):
            video_id:         str
            frame_inds:       np.ndarray(T,)   # 1基全局帧号
            key_fid:          int              # 1基，关键秒帧
            center_fid:       int              # 1基，本片段几何中心帧
            img_shape or original_shape: (H,W,...) 用于像素/归一化互转
            gt_bboxes:        (M,4)            # 关键秒所有 GT，可为像素或归一化

        输出（写回 results):
            track_vector:     torch.float32 (M,T, 2+1+K)
                              = [dx, dy, dist_water, pig_dist_1..K]
                              其中 (dx,dy) 是“平移原点后”的坐标
            water_dist:       torch.float32 (M,T,1)   # 到饮水器距离
            pig_dists:        torch.float32 (M,T,K)   # 每帧最近 K 头猪的距离（不足 K 补 0.0）
            track_tids:       List[List[int]]         # 每帧被选中的 tid（-1 表示沿用/无）
            track_center_idx: torch.long (M,)
            track_center_fid: torch.long (M,)
            center_idx_used / center_sample / center_ts  # 兼容旧字段（取第一个 ROI）
        """
        # 0) 基本元信息
        vid = results.get('video_id', 'UNKNOWN_VIDEO')
        frame_inds_1 = np.array(results['frame_inds'], dtype=np.int32)  # 1基
        T = int(frame_inds_1.shape[0])
        assert T > 0, f"{self.print_prefix} [{vid}] frame_inds 为空"

        key_fid_1 = int(results.get('key_fid', frame_inds_1[T // 2]))
        center_fid_1 = int(results.get('center_fid', frame_inds_1[T // 2]))

        img_shape = results.get('img_shape', results.get('original_shape', None))
        if img_shape is None:
            warnings.warn(f"{self.print_prefix} [{vid}] 缺少 img_shape/original_shape，默认 720x1280", stacklevel=2)
            H, W = 720, 1280
        else:
            H, W = int(img_shape[0]), int(img_shape[1])

        gt_boxes = results.get('gt_bboxes', None)  # (M,4)
        if gt_boxes is None or len(gt_boxes) == 0:
            if self.verbose:
                print(f"{self.print_prefix} [{vid}] 没有 gt_bboxes → 回退全零")
            return self._fallback(results, T, M=1)

        # 只处理一个 ROI 或全部 ROI
        if self.roi_index_key is not None and self.roi_index_key in results:
            gi = max(0, min(int(results[self.roi_index_key]), len(gt_boxes) - 1))
            roi_indices = [gi]
        else:
            roi_indices = list(range(len(gt_boxes)))  # M

        # 1) 打印输入摘要
        if self.verbose:
            print(f"{self.print_prefix} >>> Begin  LoadTrackInfo for video='{vid}'")
            print(f"{self.print_prefix}     #GT(M) = {len(roi_indices)}，T = {T}，img = {H}x{W}")
            print(f"{self.print_prefix}     frame_inds(1-based) = {frame_inds_1.tolist()}")
            print(f"{self.print_prefix}     key_fid={key_fid_1}, center_fid={center_fid_1}, offset={self.offset}")
            print(f"{self.print_prefix}     origin_alpha={self.origin_alpha}  shift=({self.shift_dx:.3f},{self.shift_dy:.3f})  water=({self.water_dx:.3f},{self.water_dy:.3f})")

        # 2) 载入 TXT（只读必要帧）
        track_file = os.path.join(self.track_base_path, f"{vid}.txt")
        if not os.path.isfile(track_file):
            if self.verbose:
                print(f"{self.print_prefix} [{vid}] 找不到跟踪文件：{track_file} → 回退全零")
            return self._fallback(results, T, M=len(roi_indices))

        needed = set(int(f) + self.offset for f in frame_inds_1)
        needed.add(int(key_fid_1) + self.offset)
        needed.add(int(center_fid_1) + self.offset)
        detections = self._load_detections(track_file, needed)  # detections[fid] = {tid: [x1,y1,x2,y2]}
        if self.verbose:
            print(f"{self.print_prefix} [{vid}] TXT loaded: file='{os.path.basename(track_file)}', #frames_read={len(detections)}")

        frames_for_lookup = (frame_inds_1 + self.offset).tolist()
        center_idx = int(np.argmin(np.abs(frame_inds_1 - center_fid_1)))

        # 3) 为每个 ROI 生成：轨迹 + 水距 + 猪邻居距
        track_vecs: List[np.ndarray] = []      # (M,T,2)  —— 注意：这是“已平移原点”的 (dx,dy)
        track_tids_list: List[List[int]] = []  # (M,T)
        used_center_idx: List[int] = []
        used_center_fid: List[int] = []

        water_dist_list: List[np.ndarray] = []  # (M,T,1)
        pig_dists_list:  List[np.ndarray] = []  # (M,T,K)

        # —— 平移后坐标系下，饮水器的位置 —— 
        # 原饮水器在 [-1,1] 为 W=(water_dx, water_dy)
        # 平移后新坐标：点' = 点 - αW，因此 W' = W - αW = (1-α)W
        water_dx_shifted = self.water_dx * (1.0 - self.origin_alpha)
        water_dy_shifted = self.water_dy * (1.0 - self.origin_alpha)

        for gi in roi_indices:
            # 3.1 GT（自动判断单位 → 像素）
            gx1, gy1, gx2, gy2 = [float(v) for v in gt_boxes[gi]]
            if max(abs(gx1), abs(gy1), abs(gx2), abs(gy2)) > 2.0:
                gt_px = [gx1, gy1, gx2, gy2]   # 已是像素
                src_tag = "px"
            else:
                gt_px = [gx1 * W, gy1 * H, gx2 * W, gy2 * H]  # 归一化→像素
                src_tag = "norm→px"

            # 清洗
            x1, y1, x2, y2 = gt_px
            x1 = max(0.0, min(x1, W - 1));  y1 = max(0.0, min(y1, H - 1))
            x2 = max(0.0, min(x2, W - 1));  y2 = max(0.0, min(y2, H - 1))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            gt_px = [x1, y1, x2, y2]

            if self.verbose:
                print(f"{self.print_prefix} [{vid}] ROI#{gi}  GT({src_tag})→px={ [round(x,1) for x in gt_px] }")

            # 3.2 起点（关键帧）
            key_fid_adj = int(key_fid_1 + self.offset)
            seed = self._choose_best_by_gt(detections.get(key_fid_adj, {}), gt_px)
            if seed is not None:
                seed_tid, seed_box, _ = seed
                prev_box = seed_box
            else:
                # 兜底：在中心帧+本段内找 IoU 最大者
                alt_frames = [int(center_fid_1 + self.offset)] + frames_for_lookup
                alt = self._seed_from_set(detections, alt_frames, gt_px)
                if alt is None:
                    # 全零输出（维度对齐）
                    track_vecs.append(np.zeros((T, 2), np.float32))
                    track_tids_list.append([-1] * T)
                    used_center_idx.append(center_idx)
                    used_center_fid.append(int(center_fid_1))
                    water_dist_list.append(np.zeros((T, 1), np.float32))
                    pig_dists_list.append(np.zeros((T, self.max_neighbors), np.float32))
                    if self.verbose:
                        print(f"{self.print_prefix} [{vid}] ROI#{gi}  Seed FAILED → zeros")
                    continue
                else:
                    _, (seed_tid, seed_box, _) = alt
                    prev_box = seed_box

            # 3.3 一步投射到 center_fid
            center_fid_adj = int(center_fid_1 + self.offset)
            proj = self._choose_best_from_prev(detections.get(center_fid_adj, {}), prev_box)
            if proj is None:
                proj_tid, proj_box = -1, prev_box
            else:
                proj_tid, proj_box = int(proj[0]), proj[1]

            # 3.4 初始化输出，写中心
            out_dxdy: List[Tuple[float, float]] = [None] * T
            tids_per_frame: List[int] = [-1] * T
            out_dxdy[center_idx] = self._box_to_dxdy_shifted(proj_box, W, H)  # ★平移后坐标
            tids_per_frame[center_idx] = proj_tid

            # 3.5 向后
            prev_box_f = proj_box
            f_match, f_miss = 0, 0
            for k in range(center_idx + 1, T):
                fid_adj = frames_for_lookup[k]
                chosen = self._choose_best_from_prev(detections.get(fid_adj, {}), prev_box_f)
                if chosen is None:
                    out_dxdy[k] = out_dxdy[k - 1]
                    tids_per_frame[k] = -1
                    f_miss += 1
                else:
                    tid, box = int(chosen[0]), chosen[1]
                    out_dxdy[k] = self._box_to_dxdy_shifted(box, W, H)  # ★平移后坐标
                    tids_per_frame[k] = tid
                    prev_box_f = box
                    f_match += 1

            # 3.6 向前
            prev_box_b = proj_box
            b_match, b_miss = 0, 0
            for k in range(center_idx - 1, -1, -1):
                fid_adj = frames_for_lookup[k]
                chosen = self._choose_best_from_prev(detections.get(fid_adj, {}), prev_box_b)
                if chosen is None:
                    out_dxdy[k] = out_dxdy[k + 1]
                    tids_per_frame[k] = -1
                    b_miss += 1
                else:
                    tid, box = int(chosen[0]), chosen[1]
                    out_dxdy[k] = self._box_to_dxdy_shifted(box, W, H)  # ★平移后坐标
                    tids_per_frame[k] = tid
                    prev_box_b = box
                    b_match += 1

            # 3.7 收集主轨迹（平移后）
            traj_xy = np.asarray(out_dxdy, dtype=np.float32)  # (T,2) —— 已平移原点
            track_vecs.append(traj_xy)
            track_tids_list.append(tids_per_frame)
            used_center_idx.append(center_idx)
            used_center_fid.append(int(center_fid_1))

            # 3.8 饮水器距离 (T,1)
            # 等价两种写法：
            #   1) 未平移：dist = ||(dx,dy) - (water_dx,water_dy)||
            #   2) 平移后：dist = ||(dx',dy') - (water_dx',water_dy')||，其中 W'=(1-α)W
            # 由于距离对平移不变，这里用“平移后”的更直观写法：
            dx = traj_xy[:, 0] - water_dx_shifted
            dy = traj_xy[:, 1] - water_dy_shifted
            dist_water = np.sqrt(dx * dx + dy * dy).reshape(T, 1).astype(np.float32)
            water_dist_list.append(dist_water)

            # 3.9 猪-猪距离 (T,K) —— 注意：其它猪的坐标也要用同一“平移后”坐标！
            K = self.max_neighbors
            pig_dists_TK = np.zeros((T, K), dtype=np.float32)
            for k in range(T):
                fid_adj = frames_for_lookup[k]
                roi_dx, roi_dy = traj_xy[k].tolist()  # 已平移
                this_tid = tids_per_frame[k]

                dists = []
                for tid_other, box in detections.get(fid_adj, {}).items():
                    if this_tid != -1 and int(tid_other) == int(this_tid):
                        continue  # 排除自己
                    # 其它猪中心（先转成未平移的 [-1,1]）
                    cx = (box[0] + box[2]) * 0.5
                    cy = (box[1] + box[3]) * 0.5
                    dx_o = (cx - W / 2.0) / (W / 2.0)
                    dy_o = (cy - H / 2.0) / (H / 2.0)
                    # 再做相同的“平移”
                    dx_o -= self.shift_dx
                    dy_o -= self.shift_dy
                    # L2
                    d = float(np.hypot(roi_dx - dx_o, roi_dy - dy_o))
                    dists.append(d)

                if len(dists) > 0:
                    dists.sort()
                    take = dists[:K]
                    if len(take) < K:
                        take += [0.0] * (K - len(take))
                else:
                    take = [0.0] * K
                pig_dists_TK[k] = np.array(take, dtype=np.float32)

            pig_dists_list.append(pig_dists_TK)

            if self.verbose:
                total_match = f_match + b_match + (1 if proj_tid != -1 else 0)
                print(f"{self.print_prefix} [{vid}] ROI#{gi}  SUMMARY: forward ✓={f_match}/✗={f_miss}, "
                      f"backward ✓={b_match}/✗={b_miss}, total_matched≈{total_match}/{T}")

        # 4) 堆叠并返回
        TV = torch.as_tensor(np.stack(track_vecs, axis=0), dtype=torch.float32)      # (M,T,2)  平移后轨迹
        WD = torch.as_tensor(np.stack(water_dist_list, axis=0), dtype=torch.float32) # (M,T,1)
        PD = torch.as_tensor(np.stack(pig_dists_list,  axis=0), dtype=torch.float32) # (M,T,K)

        track_vector = torch.cat([TV, WD, PD], dim=2)                                # (M,T,2+1+K)

        results['track_vector'] = track_vector
        results['water_dist']   = WD
        results['pig_dists']    = PD
        results['track_tids']   = track_tids_list
        results['track_center_idx'] = torch.as_tensor(np.array(used_center_idx), dtype=torch.long)
        results['track_center_fid'] = torch.as_tensor(np.array(used_center_fid), dtype=torch.long)

        # 兼容旧字段：取第一个 ROI
        results['center_idx_used'] = int(used_center_idx[0]) if used_center_idx else int(T // 2)
        results['center_sample']   = int(used_center_fid[0]) if used_center_fid else int(center_fid_1)
        results['center_ts']       = results['center_sample']

        if self.verbose:
            print(f"{self.print_prefix} [{vid}] DONE:"
                  f" track_vector={tuple(track_vector.shape)}  TV={tuple(TV.shape)} WD={tuple(WD.shape)} PD={tuple(PD.shape)}")
            print(f"{self.print_prefix} <<< End   LoadTrackInfo for video='{vid}'\n")

        return results

    # ====================== 工具函数 ====================== #
    def _load_detections(self, track_file: str, needed_frames: set) -> Dict[int, Dict[int, List[float]]]:
        """仅读 needed_frames(1基+offset)，返回 detections[fid][tid] = [x1,y1,x2,y2] 像素坐标"""
        detections: Dict[int, Dict[int, List[float]]] = {}
        with open(track_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                try:
                    fid = int(float(parts[0]))
                    if fid not in needed_frames:
                        continue
                    tid = int(float(parts[1]))
                    x1 = float(parts[2]); y1 = float(parts[3])
                    w  = float(parts[4]); h  = float(parts[5])
                    x2 = x1 + max(0.0, w); y2 = y1 + max(0.0, h)
                except Exception:
                    continue
                detections.setdefault(fid, {})[tid] = [x1, y1, x2, y2]
        return detections

    @staticmethod
    def _calculate_iou(a: List[float], b: List[float]) -> float:
        """IoU(xyxy)"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, ay2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _box_center_wh(b: List[float]) -> Tuple[float, float, float, float]:
        """返回 (cx, cy, w, h) （像素单位）"""
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                max(0.0, x2 - x1),
                max(0.0, y2 - y1))

    def _box_to_dxdy_shifted(self, box_xyxy: List[float], W: int, H: int) -> List[float]:
        """
        把像素框 → [-1,1] 相对坐标，并做“原点平移”
          旧：dx0 = (cx - W/2)/(W/2),  dy0 同理
          新：dx  = dx0 - shift_dx,    dy  = dy0 - shift_dy
        """
        cx, cy, _, _ = self._box_center_wh(box_xyxy)
        dx0 = np.clip((cx - W / 2.0) / (W / 2.0), -1.0, 1.0)
        dy0 = np.clip((cy - H / 2.0) / (H / 2.0), -1.0, 1.0)
        dx  = float(dx0 - self.shift_dx)
        dy  = float(dy0 - self.shift_dy)
        # 防守性裁剪（仍保持在 [-1,1] 左右；平移会超界一点，视需求可放宽或不裁剪）
        dx = float(np.clip(dx, -1.5, 1.5))
        dy = float(np.clip(dy, -1.5, 1.5))
        return [dx, dy]

    def _choose_best_by_gt(self, dets_in_frame: Dict[int, List[float]], gt_box: List[float]) \
            -> Optional[Tuple[int, List[float], float]]:
        """在某一帧内用 GT 选 IoU 最大的候选；若 < 门槛返回 None"""
        best = None
        best_iou = -1.0
        for tid, box in dets_in_frame.items():
            iou = self._calculate_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best = (tid, box, iou)
        if best is None or best_iou < self.iou_threshold:
            return None
        return best

    def _seed_from_set(self,
                       detections: Dict[int, Dict[int, List[float]]],
                       frame_set: List[int],
                       gt_box: List[float]) \
            -> Optional[Tuple[int, Tuple[int, List[float], float]]]:
        """在给定帧集合里找与 GT IoU 最大的一个作为起点；若 < 门槛返回 None"""
        best = None
        best_iou = -1.0
        best_fid = -1
        for fid in frame_set:
            for tid, box in detections.get(fid, {}).items():
                iou = self._calculate_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best = (tid, box, iou)
                    best_fid = fid
        if best is None or best_iou < self.iou_threshold:
            return None
        return best_fid, best

    def _choose_best_from_prev(self,
                               dets_in_frame: Dict[int, List[float]],
                               prev_box: List[float]) \
            -> Optional[Tuple[int, List[float], float, float, float]]:
        """
        在“本帧所有候选”里找“最像上一帧”的那个。
        排序 key(越大越好):
          1) IoU(prev, curr)
          2) -中心点距离^2
          3) 尺寸相似度（宽高比的乘积，越接近 1 越好）
        若最佳 IoU < follow_iou_ratio → 返回 None
        """
        if not dets_in_frame:
            return None

        pr_cx, pr_cy, _, _ = self._box_center_wh(prev_box)

        def size_ratio(a, b):
            aw = max(1e-6, a[2] - a[0]); ah = max(1e-6, a[3] - a[1])
            bw = max(1e-6, b[2] - b[0]); bh = max(1e-6, b[3] - b[1])
            rw = min(aw / bw, bw / aw)
            rh = min(ah / bh, bh / ah)
            return rw * rh

        best_item = None
        best_key = (-1.0, float('-inf'), -1.0)
        for tid, box in dets_in_frame.items():
            iou = self._calculate_iou(box, prev_box)
            cx, cy, _, _ = self._box_center_wh(box)
            d2 = (cx - pr_cx) ** 2 + (cy - pr_cy) ** 2
            sz = size_ratio(box, prev_box)
            key = (iou, -d2, sz)
            if key > best_key:
                best_key = key
                best_item = (tid, box, iou, d2, sz)

        if best_item is None:
            return None
        if best_item[2] < self.follow_iou_ratio:
            return None
        return best_item

    # ====================== 回退 ====================== #
    def _fallback(self, results: dict, T: int, M: int) -> dict:
        """回退：输出全零 (M,T, 2+1+K) 及配套字段，保证下游不崩"""
        K = self.max_neighbors
        tv = torch.zeros((M, T, 2 + 1 + K), dtype=torch.float32)
        wd = torch.zeros((M, T, 1), dtype=torch.float32)
        pd = torch.zeros((M, T, K), dtype=torch.float32)

        results['track_vector'] = tv
        results['water_dist'] = wd
        results['pig_dists'] = pd
        results['track_tids'] = [[-1] * T for _ in range(M)]

        frame_inds = np.array(results.get('frame_inds', np.arange(1, T + 1)), dtype=np.int32)
        center_idx = int(np.argmin(np.abs(frame_inds - results.get('center_fid', frame_inds[len(frame_inds)//2]))))
        center_fid = int(results.get('center_fid', frame_inds[center_idx]))

        results['center_idx_used'] = center_idx
        results['center_sample'] = center_fid
        results['center_ts'] = center_fid
        results['track_center_idx'] = torch.full((M,), center_idx, dtype=torch.long)
        results['track_center_fid'] = torch.full((M,), center_fid, dtype=torch.long)
        return results

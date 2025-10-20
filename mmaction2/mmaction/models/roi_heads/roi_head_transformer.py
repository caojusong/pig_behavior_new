from typing import List, Tuple, Union, Optional
import os
import torch
from torch import Tensor

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import bbox2roi

from mmaction.utils import ConfigType, InstanceList, SampleList

# 轨迹编码器与融合模块（确保在 config 的 custom_imports 中已注册）
from mmaction.models.roi_heads.track_modules import TrackMLPEncoder
from mmaction.models.roi_heads.scm_fusion import SCMFusionTransformer

###包含了dx,dy,+水距+k个最邻近距离
class AVARoIHead(StandardRoIHead):
    """
    AVA 风格 + SCM 融合的 RoIHead（单标签动作分类）

    训练：
      - 仍走 AVA 范式：assign + sample（常见设置 pos_fraction=1 → 只采正样本）
      - 在分类前：将视觉特征 vis_feat 与轨迹特征 traj_feat 做融合

    轨迹来源：
      - 每张图（关键秒）含有 M 个 GT（每头猪一条），
      - pipeline 里你的 LoadTrackInfo 已经写入 data_samples[i].track_vector: (M, T, D)
        * D 可以是 2（仅 dx,dy），也可以是 2+1+K（dx,dy + 水距 + K 个最近邻距离）

    推理：
      - 用 assigner 将 proposals 与 GT IoU 匹配，获得“RoI 所属的 GT 索引”，
      - 取该 GT 的轨迹编码、融合、分类

    兼容性注意：
      - MMDet 3.x 将 `pos_gt_inds` 改为 `pos_assigned_gt_inds`，这里做了兼容
      - `pos_inds` 在 3.x 仍存在；本实现使用 “采样后 bboxes 的拼接顺序” 来构造映射，更稳妥
    """

    # -------- 调试打印开关（环境变量） --------
    debug_summary = os.environ.get('AVA_SCM_DEBUG', '0') == '1'
    _printed_train_once = False
    _printed_test_once = False

    # -------- 构造：拿 track_cfg / fusion_cfg，构建编码器与融合器 --------
    def __init__(self,
                 *args,
                 track_cfg: Optional[dict] = None,
                 fusion_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # 轨迹编码器配置（可在 config 中覆写）
        track_cfg = track_cfg or dict(
            input_dim=2, hidden_dim=1024, output_dim=2304, dropout_ratio=0.5
        )
        # 融合器配置（可在 config 中覆写）
        fusion_cfg = fusion_cfg or dict(
            vis_dim=2304, traj_dim=2304, model_dim=256,
            num_layers=2, num_heads=8, dropout=0.1,
            output_dim=2304, enhanced=False, fusion_type='mean'
        )

        # 构建编码器/融合器
        self.track_encoder = TrackMLPEncoder(
            input_dim=track_cfg['input_dim'],
            hidden_dim=track_cfg['hidden_dim'],
            output_dim=track_cfg['output_dim'],
            dropout_ratio=track_cfg.get('dropout_ratio', 0.5)
        )
        self.scm_fusion = SCMFusionTransformer(
            vis_dim=fusion_cfg['vis_dim'],
            traj_dim=fusion_cfg['traj_dim'],
            model_dim=fusion_cfg['model_dim'],
            num_layers=fusion_cfg['num_layers'],
            num_heads=fusion_cfg['num_heads'],
            dropout=fusion_cfg['dropout'],
            output_dim=fusion_cfg['output_dim'],
            enhanced=fusion_cfg.get('enhanced', False),
            fusion_type=fusion_cfg.get('fusion_type', 'mean')
        )

        # 融合输出维度需匹配 bbox_head 的输入（通常 2304）
        self.fused_out_dim = fusion_cfg['output_dim']

        # 用于 _gather_tracks_for_rois 的分段信息（在 bbox_loss/predict_bbox 设置）
        self._last_num_rois_per_img: Optional[List[int]] = None

    # ---------------------- 训练入口 ----------------------
    def loss(self,
             x: Union[Tensor, Tuple[Tensor]],
             rpn_results_list: InstanceList,
             data_samples: SampleList,
             **kwargs) -> dict:
        """
        训练：前向 + 损失
        """
        assert len(rpn_results_list) == len(data_samples)

        # 收集 GT
        batch_gt_instances = [ds.gt_instances for ds in data_samples]

        # assign + sample（官方 AVA 范式：常设 pos_fraction=1）
        sampling_results: List[SamplingResult] = []
        pos_counts, neg_counts = [], []
        for i in range(len(data_samples)):
            rpn_results = rpn_results_list[i]
            # 统一字段：mmdet 标准用 priors 表示 proposals/bboxes
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i], None
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result, rpn_results, batch_gt_instances[i]
            )
            sampling_results.append(sampling_result)
            # 统计
            pos_counts.append(len(getattr(sampling_result, 'pos_inds', [])))
            neg_counts.append(len(getattr(sampling_result, 'neg_inds', [])))

        if self.debug_summary and not AVARoIHead._printed_train_once:
            print(f"[SCM-ROIHead][Train] imgs={len(data_samples)} "
                  f"pos={sum(pos_counts)} neg={sum(neg_counts)} "
                  f"pos/img={pos_counts} neg/img={neg_counts}", flush=True)
            AVARoIHead._printed_train_once = True

        batch_img_metas = [ds.metainfo for ds in data_samples]

        # 前向 + 融合 + 损失
        losses = dict()
        bbox_results = self.bbox_loss(
            x, sampling_results, batch_img_metas, data_samples=data_samples
        )
        losses.update(bbox_results['loss_bbox'])
        return losses

    # ---------------------- 通用前向（提取 RoI 特征） ----------------------
    def _bbox_forward(self,
                      x: Union[Tensor, Tuple[Tensor]],
                      rois: Tensor,
                      batch_img_metas: List[dict],
                      **kwargs) -> dict:
        """
        RoIAlign + bbox_head 前的特征提取。
        在本实现里，我们先提特征与 GAP，真正的融合放到 bbox_loss / predict_bbox 里处理。
        """
        bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(
                bbox_feats, feat=global_feat, rois=rois, img_metas=batch_img_metas
            )
        return dict(bbox_feats=bbox_feats)  # (N, C, T, H, W)

    # ---------------------- 训练：融合 + 损失 ----------------------
    def bbox_loss(self,
                  x: Union[Tensor, Tuple[Tensor]],
                  sampling_results: List[SamplingResult],
                  batch_img_metas: List[dict],
                  data_samples: SampleList,
                  **kwargs) -> dict:
        """
        训练路径：
          1) proposals -> rois
          2) RoI 特征 -> (N,C,T,H,W) -> GAP -> (N,C) = vis_feat
          3) 通过 sampling_results 构造 "rois -> gt_idx" 的映射
          4) 从 data_samples[i].track_vector[gt_idx] 取 (T,D) → 编码 → SCM 融合
          5) 融合结果 reshape 回 (N,C,1,1,1) 给 bbox_head 分类
          6) 用 bbox_head.loss_and_target() (与官方 AVA 一致) 计算损失
        """
        # 1) proposals -> rois（用采样后的 bboxes）
        rois = bbox2roi([res.bboxes for res in sampling_results])  # (N,5)

        # 显式记录每张图的 RoI 数量，供 _gather_tracks_for_rois 分段使用
        num_per_img = [len(res.bboxes) for res in sampling_results]
        self._set_num_rois_per_img(num_per_img)

        # 2) 提特征 + GAP
        fwd = self._bbox_forward(x, rois, batch_img_metas)
        bbox_feats = fwd['bbox_feats']                 # (N, C, T, H, W)
        N, C = bbox_feats.shape[:2]
        vis_feat = bbox_feats.mean(dim=[2, 3, 4])      # (N, C)

        # 3) rois -> gt_idx（-1 表示非正样本）
        roi_to_gt = self._map_rois_to_gt_via_sampling(sampling_results)  # (N,)

        # 4) 取轨迹 -> 维度护栏 -> 编码 -> 融合
        track_vec_rois = self._gather_tracks_for_rois(roi_to_gt, data_samples, rois.device)  # (N, T, D)
        D_in = int(track_vec_rois.shape[-1])
        enc_in = int(self.track_encoder.linear1.in_features)
        assert D_in == enc_in, (
            f"[Track-Input-Dim Mismatch] track_vec last dim={D_in}, "
            f"but TrackMLPEncoder.input_dim={enc_in}. "
            f"请在 config 的 track_cfg.input_dim 设置为 {D_in}（=2 或 2+1+K）。"
        )

        traj_feat = self.track_encoder(track_vec_rois)                                       # (N, C')
        fused = self.scm_fusion(vis_feat, traj_feat)                                         # (N, fused_dim)

        # 5) reshape 回 5D，交 bbox_head 分类
        fused_5d = fused.view(N, fused.shape[1], 1, 1, 1)  # (N, C, 1, 1, 1)
        cls_score = self.bbox_head(fused_5d)

        # 6) loss（由 bbox_head 内部根据 sampling_results 的正负样本选择参与损失的 RoI）
        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=cls_score,
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg
        )

        if self.debug_summary and not AVARoIHead._printed_train_once:
            print(f"[SCM-ROIHead][Train] vis={tuple(vis_feat.shape)} "
                  f"traj_in={tuple(track_vec_rois.shape)} traj={tuple(traj_feat.shape)} "
                  f"fused={tuple(fused.shape)} cls={tuple(cls_score.shape)}",
                  flush=True)
            AVARoIHead._printed_train_once = True

        return dict(loss_bbox=bbox_loss_and_target['loss_bbox'])

    # ---------------------- 推理：融合 + 预测 ----------------------
    def predict(self,
                x: Union[Tensor, Tuple[Tensor]],
                rpn_results_list: InstanceList,
                data_samples: SampleList,
                **kwargs) -> InstanceList:
        """
        推理：保持与官方一致（batch=1）
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [ds.metainfo for ds in data_samples]
        if isinstance(x, tuple):
            x_shape = x[0].shape
        else:
            x_shape = x.shape
        assert x_shape[0] == 1, 'only accept 1 sample at test mode'
        assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

        results_list = self.predict_bbox(
            x, batch_img_metas, rpn_results_list,
            rcnn_test_cfg=self.test_cfg, data_samples=data_samples
        )
        return results_list

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     data_samples: SampleList) -> InstanceList:
        """
        推理路径：
          - proposals -> rois
          - 特征 -> vis_feat
          - assigner 做一次匹配（只用来得到“RoI 属于哪头猪”），取轨迹 -> 融合 -> 分类
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)                                    # (N,5)

        # 设置每张图的 RoI 数量，供 _gather_tracks_for_rois 使用
        num_per_img = [len(p) for p in proposals]
        self._set_num_rois_per_img(num_per_img)

        # 1) 提特征 + GAP
        fwd = self._bbox_forward(x, rois, batch_img_metas)
        bbox_feats = fwd['bbox_feats']                                 # (N,C,T,H,W)
        N, C = bbox_feats.shape[:2]
        vis_feat = bbox_feats.mean(dim=[2, 3, 4])                      # (N,C)

        # 2) 测试期：assigner 匹配（得到 rois -> gt_idx）
        roi_to_gt = self._match_rois_to_gt_at_test(proposals, data_samples)  # (N,) or -1

        # 3) 取轨迹 -> 维度护栏 -> 编码 -> 融合 -> 分类
        track_vec_rois = self._gather_tracks_for_rois(roi_to_gt, data_samples, rois.device)  # (N,T,D)
        D_in = int(track_vec_rois.shape[-1])
        enc_in = int(self.track_encoder.linear1.in_features)
        assert D_in == enc_in, (
            f"[Track-Input-Dim Mismatch] track_vec last dim={D_in}, "
            f"but TrackMLPEncoder.input_dim={enc_in}. "
            f"请在 config 的 track_cfg.input_dim 设置为 {D_in}（=2 或 2+1+K）。"
        )

        traj_feat = self.track_encoder(track_vec_rois)
        fused = self.scm_fusion(vis_feat, traj_feat)
        fused_5d = fused.view(N, fused.shape[1], 1, 1, 1)
        cls_score = self.bbox_head(fused_5d)

        # 4) 拆回每张图并后处理
        rois_split = rois.split(num_per_img, 0)
        cls_split = cls_score.split(num_per_img, 0)
        result_list = self.bbox_head.predict_by_feat(
            rois=rois_split,
            cls_scores=cls_split,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg
        )

        if self.debug_summary and not AVARoIHead._printed_test_once:
            print(f"[SCM-ROIHead][Test] vis={tuple(vis_feat.shape)} "
                  f"traj_in={tuple(track_vec_rois.shape)} traj={tuple(traj_feat.shape)} "
                  f"fused={tuple(fused.shape)} cls={tuple(cls_score.shape)}",
                  flush=True)
            AVARoIHead._printed_test_once = True

        return result_list

    # ================= 工具：训练期 RoI -> GT 映射（基于采样后顺序） =================
    def _map_rois_to_gt_via_sampling(self, sampling_results: List[SamplingResult]) -> torch.Tensor:
        """
        基于“采样后的 bboxes 顺序”构造 RoI -> GT 映射：
        每张图：res.bboxes = cat(pos_bboxes, neg_bboxes)
        因此本图段前 n_pos 个 RoI 是正样本，其 GT 索引为 res.pos_gt_inds / pos_assigned_gt_inds，
        其余为 -1。
        """
        # 每图 RoI 数（采样后）
        num_per_img = [int(res.bboxes.size(0)) for res in sampling_results]
        self._set_num_rois_per_img(num_per_img)

        total = sum(num_per_img)
        device = sampling_results[0].bboxes.device if len(sampling_results) else torch.device('cpu')
        roi_to_gt = torch.full((total,), -1, dtype=torch.long, device=device)

        base = 0
        for i, res in enumerate(sampling_results):
            # 正样本数量与 GT 索引
            n_pos = int(res.pos_bboxes.size(0)) if hasattr(res, 'pos_bboxes') else 0
            pos_gt_inds = getattr(res, 'pos_gt_inds', None)
            if pos_gt_inds is None:
                pos_gt_inds = getattr(res, 'pos_assigned_gt_inds', None)

            if n_pos > 0:
                assert pos_gt_inds is not None, 'SamplingResult 缺少 pos_gt_inds / pos_assigned_gt_inds'
                pos_gt_inds = torch.as_tensor(pos_gt_inds, dtype=torch.long, device=device).view(-1)
                if pos_gt_inds.numel() != n_pos:
                    assert pos_gt_inds.numel() >= n_pos, \
                        f'pos_gt_inds.len={pos_gt_inds.numel()} < n_pos={n_pos}'
                    pos_gt_inds = pos_gt_inds[:n_pos]
                roi_to_gt[base:base + n_pos] = pos_gt_inds
            # 负样本保持为 -1
            base += num_per_img[i]

        return roi_to_gt

    # ================= 工具：推理期 RoI -> GT 映射（assigner 一次） =================
    def _match_rois_to_gt_at_test(self,
                                  proposals: List[Tensor],
                                  data_samples: SampleList) -> torch.Tensor:
        """
        测试期没有 sampling_results，这里用 assigner 对 proposals 与 GT 做一次 IoU 匹配，
        返回每个 RoI 对应的 GT 索引（-1 表示没匹配到）。
        """
        assert len(proposals) == len(data_samples)
        num_per_img = [len(p) for p in proposals]
        total = sum(num_per_img)
        device = proposals[0].device if len(proposals) and proposals[0].numel() > 0 else torch.device('cpu')
        out = torch.full((total,), -1, dtype=torch.long, device=device)

        base = 0
        for i, (props, ds) in enumerate(zip(proposals, data_samples)):
            # 构造与训练一致的输入容器
            inst = type('Tmp', (), {})()
            inst.priors = props
            gt = ds.gt_instances

            assign_result = self.bbox_assigner.assign(inst, gt, None)
            # MMDet 中 assign_result.gt_inds: 1..M 为正样本，0 为背景
            assigned = assign_result.gt_inds.long()  # shape (num_props,)
            mapped = assigned - 1                    # → -1/0..M-1
            out[base:base + len(props)] = mapped
            base += len(props)

        return out

    # ================= 工具：按 RoI 取轨迹 (N,T,D) =================
    def _gather_tracks_for_rois(self,
                                roi_to_gt: torch.Tensor,
                                data_samples: SampleList,
                                device: torch.device) -> torch.Tensor:
        """
        把每个 RoI 对应的 GT 轨迹取出来，拼成 (N,T,D)。

        - data_samples[i].track_vector: (M_i, T, D)（LoadTrackInfo 事先写入；D=2 或 2+1+K）
        - roi_to_gt: 全 batch 拼接后的 RoI -> GT 映射（-1 表示没有匹配）
        - 需要 self._last_num_rois_per_img 告知每张图 RoI 数，从而进行分段
        """
        roi_to_gt = roi_to_gt.to(device).view(-1)
        num_per_img = getattr(self, "_last_num_rois_per_img", None)
        assert num_per_img is not None, "Internal error: num_per_img not set for track gather."

        # 整理每张图的 (M_i, T, D)，并拼成大表，记录偏移
        per_img_tracks = []
        offsets = []
        cur = 0
        T = None
        D = None
        for ds in data_samples:
            tv = ds.track_vector  # 可能是 torch 或 numpy；形如 (M_i, T, D)
            if not torch.is_tensor(tv):
                tv = torch.as_tensor(tv, dtype=torch.float32)
            if T is None:
                T = int(tv.shape[1])
            if D is None:
                D = int(tv.shape[2])
            per_img_tracks.append(tv)
            m = tv.shape[0]
            offsets.append((cur, cur + m))
            cur += m

        if len(per_img_tracks) > 0:
            big_tracks = torch.cat(per_img_tracks, dim=0).to(device)  # (sum M_i, T, D)
        else:
            T = 1 if T is None else T
            D = 2 if D is None else D
            big_tracks = torch.zeros((0, T, D), dtype=torch.float32, device=device)

        N = roi_to_gt.numel()
        out = torch.zeros((N, T, D), dtype=torch.float32, device=device)

        base = 0
        for i, n in enumerate(num_per_img):
            l, r = offsets[i]                 # 第 i 张图的 GT 在 big_tracks 的区间 [l, r)
            cur_slice = slice(base, base + n) # 第 i 张图对应的 RoI 段
            gt_idx_local = roi_to_gt[cur_slice]  # (n,) 局部 GT 索引（0..M_i-1 或 -1）

            valid = gt_idx_local >= 0
            if valid.any():
                full_idx = l + gt_idx_local[valid]
                idx_full = torch.arange(n, device=device) + base  # 当前图段的全局行号
                idx_full = idx_full[valid]
                out[cur_slice][valid] = big_tracks[full_idx]

            base += n

        return out

    # ============== 上游设置：记录每张图 RoI 数（训练/推理各设置一次） ==============
    def _set_num_rois_per_img(self, num_per_img: List[int]):
        """
        由 bbox_loss() / predict_bbox() 调用：
        告诉 _gather_tracks_for_rois：拼接后的 rois 如何按“每张图段落”切分。
        """
        self._last_num_rois_per_img = list(map(int, num_per_img))

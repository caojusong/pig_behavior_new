# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import bbox2roi
from torch import Tensor

from mmaction.utils import ConfigType, InstanceList, SampleList
from mmaction.models.roi_heads.track_modules import TrackMLPEncoder
from mmaction.models.roi_heads.scm_fusion import SCMFusionTransformer

class AVARoIHead(StandardRoIHead):
    
    def __init__(self, *args,
                 track_cfg=dict(input_dim=3, hidden_dim=1024,
                                output_dim=2304, dropout_ratio=0.5),
                 fusion_cfg=dict(vis_dim=2304,
                                 traj_dim=2304,
                                 model_dim=256,
                                 num_layers=4,
                                 num_heads=8,
                                 dropout=0.1,
                                 fusion_type='mean',
                                 enhanced=True,
                                 output_dim=2304),
                  **kwargs):
        # 1) 先调用父类构造，完成 nn.Module 初始化
        super().__init__(*args, **kwargs)
        # 初始化轨迹编码器和融合模块
        self.track_encoder = TrackMLPEncoder(
            input_dim=track_cfg['input_dim'],
            hidden_dim=track_cfg['hidden_dim'],
            output_dim=track_cfg['output_dim'],
            dropout_ratio=track_cfg['dropout_ratio']
        )
        self.scm_fusion = SCMFusionTransformer(
            vis_dim=fusion_cfg['vis_dim'],
            traj_dim=fusion_cfg['traj_dim'],
            model_dim=fusion_cfg['model_dim'],
            num_layers=fusion_cfg['num_layers'],
            num_heads=fusion_cfg['num_heads'],
            dropout=fusion_cfg['dropout'],
            output_dim=fusion_cfg['output_dim'],
            enhanced=fusion_cfg['enhanced'],
            fusion_type=fusion_cfg['fusion_type']
        )

    def loss(self, x: Union[Tensor,
                            Tuple[Tensor]], rpn_results_list: InstanceList,
             data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (Tensor or Tuple[Tensor]): The image features extracted by
                the upstream network.
            rpn_results_list (List[:obj:`InstanceData`]): List of region
                proposals.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(rpn_results_list) == len(data_samples)
        batch_gt_instances = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)

        # assign gts and sample proposals
        num_imgs = len(data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(rpn_results,
                                                      batch_gt_instances[i],
                                                      None)
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       rpn_results,
                                                       batch_gt_instances[i])
            sampling_results.append(sampling_result)

        # LFB needs meta_info: 'img_key'
        batch_img_metas = [
            data_samples.metainfo for data_samples in data_samples
        ]

        losses = dict()
        # bbox head forward and loss
        bbox_results = self.bbox_loss(x, sampling_results,
                                     batch_img_metas,
                                     data_samples=data_samples)
        losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x: Union[Tensor, Tuple[Tensor]], rois: Tensor,
                      batch_img_metas: List[dict], **kwargs) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (Tensor or Tuple[Tensor]): The image features extracted by
                the upstream network.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            batch_img_metas (List[dict]): List of image information.

        Returns:
                dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(
                bbox_feats,
                feat=global_feat,
                rois=rois,
                img_metas=batch_img_metas)

        cls_score = self.bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self,
                  x: Union[Tensor, Tuple[Tensor]],
                  sampling_results: List[SamplingResult],
                  batch_img_metas: List[dict],
                  data_samples: SampleList,
                  **kwargs) -> dict:
        """重写：在训练时将轨迹融合进 RoI 特征再计算 loss."""
        # 1. proposals -> rois
        rois = bbox2roi([res.priors for res in sampling_results])

        # 2. RoIAlign 提取特征
        bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(
                bbox_feats, feat=global_feat,
                rois=rois, img_metas=batch_img_metas
            )

        # 3. squeeze 空间/时间到 (N, C)
        N, C = bbox_feats.shape[:2]
        vis_feat = bbox_feats.mean(dim=[2, 3, 4])  # (N, C)  # (N, C)

        # 4. 取轨迹向量 (B, T, 2)
        track_vecs = torch.stack([
            ds.track_vector.to(vis_feat.device)                         # 已是 Tensor
            if isinstance(ds.track_vector, torch.Tensor)                # 判断
            else torch.from_numpy(ds.track_vector).to(vis_feat.device)  # 仍是 ndarray
            for ds in data_samples
        ], dim=0)
        traj_feat = self.track_encoder(track_vecs)    # (B, C)
        batch_inds = rois[:, 0].long()                # (N,)
        traj_feat = traj_feat[batch_inds]             # (N, C)

        # 7. Transformer 融合 → SCMFusion 内部已经拼接了 is_near_ratio，
        fused_feat = self.scm_fusion(vis_feat, traj_feat)
        # 8. reshape 成 (N, C,1,1,1)，这里 C==output_dim+1
        fused_feats = fused_feat.view(N, fused_feat.shape[1], 1, 1, 1)
        # 9. 分类 loss
        cls_score = self.bbox_head(fused_feats).view(N, -1)
        loss_and_target = self.bbox_head.loss_and_target(
            cls_score=cls_score,
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg
        )
        return dict(loss_bbox=loss_and_target['loss_bbox'])

    def predict(self, x: Union[Tensor,
                               Tuple[Tensor]], rpn_results_list: InstanceList,
                data_samples: SampleList, **kwargs) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (Tensor or Tuple[Tensor]): The image features extracted by
                the upstream network.
            rpn_results_list (List[:obj:`InstanceData`]): list of region
                proposals.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            List[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in data_samples
        ]
        if isinstance(x, tuple):
            x_shape = x[0].shape
        else:
            x_shape = x.shape

        assert x_shape[0] == 1, 'only accept 1 sample at test mode'
        assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

        # 传入 data_samples
        results_list = self.predict_bbox(
            x, batch_img_metas, rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            data_samples=data_samples)

        return results_list

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     data_samples: SampleList) -> InstanceList:
        """推理时融合轨迹特征再预测。"""
        # 1. proposals -> rois
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        # 2. RoIAlign
        bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(
                bbox_feats, feat=global_feat,
                rois=rois, img_metas=batch_img_metas)

        # 3. squeeze -> (N, C)
        N, C = bbox_feats.shape[:2]
        vis_feat = bbox_feats.view(N, C, -1).mean(-1)  # (N, C)

        # 4. 取轨迹向量 (B, T, 2)
        track_vecs = torch.stack([
            ds.track_vector.to(vis_feat.device)
            if isinstance(ds.track_vector, torch.Tensor)
            else torch.from_numpy(ds.track_vector).to(vis_feat.device)
            for ds in data_samples
        ], dim=0)

        # 5. MLP 编码 -> (B, C)
       #方法一：零拷贝按 batch 索引精确对齐，显存更省。
        traj_feat = self.track_encoder(track_vecs)    # (B, C)
        batch_inds = rois[:, 0].long()                # (N,)
        traj_feat = traj_feat[batch_inds]             # (N, C)
        #6、
        fused_feat = self.scm_fusion(vis_feat, traj_feat)
        # 7. reshape 并分类
        fused_feats = fused_feat.view(N, fused_feat.shape[1], 1, 1, 1)
        cls_score = self.bbox_head(fused_feats)
        # 8. split 回每张图 & 返回
        num_per_img = tuple(len(p) for p in proposals)
        rois_split = rois.split(num_per_img, 0)
        cls_split = cls_score.split(num_per_img, 0)

        return self.bbox_head.predict_by_feat(
            rois=rois_split,
            cls_scores=cls_split,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg
        )



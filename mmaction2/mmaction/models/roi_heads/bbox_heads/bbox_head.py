# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.task_modules.samplers import SamplingResult
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
# Resolve cross-entropy function to support multi-target in Torch < 1.10
#   This is a very basic 'hack', with minimal functionality to support the
#   procedure under prior torch versions
from packaging import version as pv
from torch import Tensor

from mmaction.structures.bbox import bbox_target
from mmaction.utils import InstanceList

if pv.parse(torch.__version__) < pv.parse('1.10'):

    def cross_entropy_loss(input, target, reduction='None'):
        input = input.log_softmax(dim=-1)  # Compute Log of Softmax
        loss = -(input * target).sum(dim=-1)  # Compute Loss manually
        if reduction.lower() == 'mean':
            return loss.mean()
        elif reduction.lower() == 'sum':
            return loss.sum()
        else:
            return loss
else:
    cross_entropy_loss = F.cross_entropy


class BBoxHeadAVA(nn.Module):
    """Simplest RoI head, with only one fc layer for classification.

 Args:
        background_class (bool): 是否将类别0设为背景类,并在计算损失时忽略该类别。
        temporal_pool_type (str): 时间维度的池化类型，可选值为``avg``（平均池化）或``max``（最大池化），默认使用平均池化（``avg``）。
        spatial_pool_type (str): 空间维度的池化类型，可选值为``avg``（平均池化）或``max``（最大池化），默认使用最大池化（``max``）。
        in_channels (int): 输入特征的通道数,默认值为2048。
        focal_alpha (float): Focal Loss的超参数α(阿尔法)。当``alpha=1``且``gamma=0``时，Focal Loss退化为带Sigmoid的二分类交叉熵损失（BCELossWithLogits），默认值为1。
        focal_gamma (float): Focal Loss的超参数γ（伽马）。当``alpha=1``且``gamma=0``时，Focal Loss退化为BCELossWithLogits，默认值为0。
        num_classes (int): 目标分类的类别数，默认值为81。
        dropout_ratio (float): Dropout层的丢弃比例（取值范围``[0, 1]``），用于防止过拟合。默认值为0（不启用Dropout）。
        dropout_before_pool (bool): 是否在时空池化（spatial temporal pooling）前应用Dropout。默认值为True（在池化前丢弃部分特征）。
        topk (int 或 Tuple[int]): 用于评估Top-K准确率的参数。默认值为``(3, 5)``（即计算前3和前5正确的准确率）。
        multilabel (bool): 是否用于多标签分类任务（一个样本可能属于多个类别）。默认值为True（支持多标签）。
        mlp_head (bool): 是否使用MLP（多层感知机）作为分类头。若为False（默认），则仅用单个线性层（``nn.Linear``若为True,则使用两层线性层带ReLU激活。
    """

    def __init__(
            self,
            background_class: bool,
            temporal_pool_type: str = 'avg',
            spatial_pool_type: str = 'max',
            in_channels: int = 2048,
            focal_gamma: float = 0.,
            focal_alpha: float = 1.,
            num_classes: int = 81,  # First class reserved (BBox as pos/neg)
            dropout_ratio: float = 0,
            dropout_before_pool: bool = True,
            topk: Union[int, Tuple[int]] = (3, 5),
            multilabel: bool = False,
            mlp_head: bool = False) -> None:
        super(BBoxHeadAVA, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        self.background_class = background_class

        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk, )
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
            self.topk = topk
        else:
            raise TypeError('topk should be int or tuple[int], '
                            f'but get {type(topk)}')
        # Class 0 is ignored when calculating accuracy,
        #      so topk cannot be equal to num_classes.
        assert all([k < num_classes for k in self.topk])

        in_channels = self.in_channels
        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        # in_channels 已经包括了 is_near_ratio 那一维，所以这里直接用 in_channels
        in_ch = in_channels
        if mlp_head:
             self.fc_cls = nn.Sequential(
                 nn.Linear(in_ch, in_ch), nn.ReLU(),
                 nn.Linear(in_ch, num_classes))
        else:
             self.fc_cls = nn.Linear(in_ch, num_classes)
    def init_weights(self) -> None:
        """Initialize the classification head."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the classification logits given ROI features."""
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = self.temporal_pool(x)
        x = self.spatial_pool(x)

        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)      
        return cls_score

    @staticmethod
    def get_targets(sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict) -> tuple:
        pos_proposals = [res.pos_priors for res in sampling_results]
        neg_proposals = [res.neg_priors for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_targets = bbox_target(pos_proposals, neg_proposals, pos_gt_labels,
                                  rcnn_train_cfg)
        return cls_targets

    @staticmethod
    def get_recall_prec(pred_vec: Tensor, target_vec: Tensor) -> tuple:
        """Computes the Recall/Precision for both multi-label and single label
        scenarios.

        Note that the computation calculates the micro average.

        Note, that in both cases, the concept of correct/incorrect is the same.
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1 - for
                single label it is expected that only one element is on (1)
                although this is not enforced.
        """
        correct = pred_vec & target_vec
        recall = correct.sum(1) / target_vec.sum(1).float()  # Enforce Float
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    @staticmethod
    def topk_to_matrix(probs: Tensor, k: int) -> Tensor:
        """Converts top-k to binary matrix."""
        topk_labels = probs.topk(k, 1, True, True)[1]
        topk_matrix = probs.new_full(probs.size(), 0, dtype=torch.bool)
        for i in range(probs.shape[0]):
            topk_matrix[i, topk_labels[i]] = 1
        return topk_matrix

    def topk_accuracy(self,
                      pred: Tensor,
                      target: Tensor,
                      thr: float = 0.5) -> tuple:
        """Computes the Top-K Accuracies for both single and multi-label
        scenarios."""
        # Define Target vector:
        target_bool = target > 0.5

        # Branch on Multilabel for computing output classification
        if self.multilabel:
            pred = pred.sigmoid()
        else:
            pred = pred.softmax(dim=1)

        # Compute at threshold (K=1 for single)
        if self.multilabel:
            pred_bool = pred > thr
        else:
            pred_bool = self.topk_to_matrix(pred, 1)
        recall_thr, prec_thr = self.get_recall_prec(pred_bool, target_bool)

        # Compute at various K
        recalls_k, precs_k = [], []
        for k in self.topk:
            pred_bool = self.topk_to_matrix(pred, k)
            recall, prec = self.get_recall_prec(pred_bool, target_bool)
            recalls_k.append(recall)
            precs_k.append(prec)

        # Return all
        return recall_thr, prec_thr, recalls_k, precs_k

    def loss_and_target(self, cls_score: Tensor, rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_targets = self.get_targets(sampling_results, rcnn_train_cfg)
        labels, _ = cls_targets
        losses = dict()          #修改为单标签，多标签返回返回的是 one-hot 向量（多标签格式）
        # Only use the cls_score
        if cls_score is not None:
            if self.background_class:
                labels = labels[:, 1:]  # Get valid labels (ignore first one)
                cls_score = cls_score[:, 1:]
            pos_inds = torch.sum(labels, dim=-1) > 0
            cls_score = cls_score[pos_inds]
            labels = labels[pos_inds]

            # Compute First Recall/Precisions
            #   This has to be done first before normalising the label-space.
            recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]

            # If Single-label, need to ensure that target labels sum to 1: ie
            #   that they are valid probabilities.
            if not self.multilabel:
                # 1) 先把 one‑hot → 整数索引，给 CE 用
                labels_index = labels.argmax(dim=1)          # (N,)

                # 2) 交叉熵 +（可选）Focal 变体
                loss = cross_entropy_loss(cls_score, labels_index, reduction='none')
                pt = torch.exp(-loss)
                F_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * loss
                losses['loss_action_cls'] = F_loss.mean()
                # 3) 为了算 top‑k / AVAMetric，再恢复 one‑hot 版
                #    用 cls_score 的通道数（background_class=True 后是 num_classes-1）
                num_pred_classes = cls_score.size(1)
                labels_for_metric = F.one_hot(labels_index,
                                              num_classes=num_pred_classes).float()
                # 计算指标
                recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(
                    cls_score, labels_for_metric, thr=0.5)
                losses['recall@thr=0.5'] = recall_thr
                losses['prec@thr=0.5'] = prec_thr
                for i, k in enumerate(self.topk):
                    losses[f'recall@top{k}'] = recall_k[i]
                    losses[f'prec@top{k}']  = prec_k[i]

                # 把 one‑hot 版传回，供 AVAMetric 使用
                labels = labels_for_metric
#原版
# Select Loss function based on single/multi-label
#   NB. Both losses auto-compute sigmoid/softmax on prediction
#if self.multilabel:
#    loss_func = F.binary_cross_entropy_with_logits
#else:
#    loss_func = cross_entropy_loss

# Compute loss
#loss = loss_func(cls_score, labels, reduction='none')
#pt = torch.exp(-loss)
#F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
#losses['loss_action_cls'] = torch.mean(F_loss)

        return dict(loss_bbox=losses, bbox_targets=cls_targets)

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                img_meta=img_meta,
                rcnn_test_cfg=rcnn_test_cfg,
                **kwargs)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(self,
                                roi: Tensor,
                                cls_score: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: Optional[ConfigDict] = None,
                                **kwargs) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        # Handle Multi/Single Label
        if cls_score is not None:
            if self.multilabel:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(dim=-1)
        else:
            scores = None

        bboxes = roi[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_meta['img_shape']
        if img_meta.get('flip', False):
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        crop_quadruple = img_meta.get('crop_quadruple', np.array([0, 0, 1, 1]))
        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)

        results.bboxes = bboxes
        results.scores = scores

        return results

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
import torch.distributed as dist  # 文件顶部如果没有就加
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
        focal_alpha (float): Focal Loss的超参数(阿尔法)。当``alpha=1``且``gamma=0``时,Focal Loss退化为带Sigmoid的二分类交叉熵损失(BCELossWithLogits),默认值为1。
        focal_gamma (float): Focal Loss的超参数(伽马)。当``alpha=1``且``gamma=0``时,Focal Loss退化为BCELossWithLogits,默认值为0。
        num_classes (int): 目标分类的类别数,默认值为81。
        dropout_ratio (float): Dropout层的丢弃比例(取值范围``[0, 1]``),用于防止过拟合。默认值为0(不启用Dropout)。
        dropout_before_pool (bool): 是否在时空池化(spatial temporal pooling)前应用Dropout。默认值为True(在池化前丢弃部分特征)。
        topk (int 或 Tuple[int]): 用于评估Top-K准确率的参数。默认值为``(3, 5)``(即计算前3和前5正确的准确率)。
        multilabel (bool): 是否用于多标签分类任务(一个样本可能属于多个类别)。默认值为True(支持多标签）。
        mlp_head (bool): 是否使用MLP(多层感知机)作为分类头。若为False(默认)，则仅用单个线性层（``nn.Linear``若为True,则使用两层线性层带ReLU激活。
    """

    def __init__(
            self,
            background_class: bool,
            temporal_pool_type: str = 'avg',
            spatial_pool_type: str = 'max',
            in_channels: int = 2048,
            focal_gamma: float = 0.,
            focal_alpha: float = 1.,
            num_classes: int = 6,  # First class reserved (BBox as pos/neg)
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

    def forward(self, x: Tensor) -> Tensor:   #(时域池化) → (空域池化) → (展平) → (全连接分类)。
        """Computes the classification logits given ROI features."""
        if self.dropout_before_pool and self.dropout_ratio > 0:   # 检查是否需要在池化前应用 Dropout，并且 Dropout 比例大于 0
            x = self.dropout(x)

        x = self.temporal_pool(x)                                # 对输入特征 x 进行时间维度的池化操作
        x = self.spatial_pool(x)                                 # 对经过时间池化后的特征 x 进行空间维度的池化操作

        if not self.dropout_before_pool and self.dropout_ratio > 0: 
            x = self.dropout(x)

        x = x.view(x.size(0), -1)                               # 将特征 x 展平为二维张量，第一维为批次大小，第二维为剩余维度的乘积
        cls_score = self.fc_cls(x)      
        return cls_score

    @staticmethod
    def get_targets(sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict) -> tuple:
        pos_proposals = [res.pos_priors for res in sampling_results]    #使用列表推导式从 sampling_results 列表中提取每个 SamplingResult 对象的正样本建议框（pos_priors），最终得到一个包含所有正样本建议框的列表。
        neg_proposals = [res.neg_priors for res in sampling_results]    #提取每个 SamplingResult 对象的负样本建议框（neg_priors），得到一个包含所有负样本建议框的列表。
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results] #提取每个 SamplingResult 对象的正样本对应的真实标签（pos_gt_labels），得到一个包含所有正样本真实标签的列表
        cls_targets = bbox_target(pos_proposals, neg_proposals, pos_gt_labels,
                                  rcnn_train_cfg)
        return cls_targets

    @staticmethod
    def get_recall_prec(pred_vec: Tensor, target_vec: Tensor) -> tuple:    #该方法用于计算多标签和单标签场景下的召回率（Recall）和精确率（Precision），采用的是微观平均（micro average）的计算方式
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
    def topk_to_matrix(probs: Tensor, k: int) -> Tensor:    #主要功能是将概率矩阵 probs 转换为一个二进制矩阵，该矩阵标记出每个样本概率最高的前 k 个类别
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
        """Computes the Top-K Accuracies for both single and multi-label           #定义了 BBoxHeadAVA 类中的 topk_accuracy 方法，该方法用于计算单标签和多标签场景下的 Top-K 召回率和精确率
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
                        rcnn_train_cfg: ConfigDict, **kwargs) -> dict:        #该方法的主要功能是基于边界框头提取的特征计算损失，并生成分类目标
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score:分类预测结果张量,形状为 (batch_size * num_proposals_single_image, num_classes)。
            rois:感兴趣区域(RoIs)张量,形状为 (batch_size * num_proposals_single_image, 5)，第一列表示每个 RoI 所属的批次 ID。
            sampling_results:一个包含 SamplingResult 对象的列表，代表批量中所有图像采样后的分配结果。
            rcnn_train_cfg:RCNN 的训练配置对象。
            **kwargs:可变关键字参数，但在方法中未被使用。
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_targets = self.get_targets(sampling_results, rcnn_train_cfg)
        labels, _ = cls_targets
        # —— 新增：打印本批次里到底有哪些标签 ID ——  
        #print("Unique labels in this batch:", labels.unique().cpu().tolist())
        losses = dict()          #修改为单标签，多标签返回返回的是 one-hot 向量（多标签格式）
        # Only use the cls_score
        # 1) 如果你开启了 background_class，那么先把背景通道抹掉
        if cls_score is not None:
            if self.background_class:
                labels = labels[:, 1:]  # Get valid labels (ignore first one)
                cls_score = cls_score[:, 1:]
            # 2) 只对至少一个前景为 1 的行计算（丢掉全 0 的背景 ROI）
            pos_inds = torch.sum(labels, dim=-1) > 0     #创建一个布尔索引，标记哪些样本是正样本（标签中至少有一个类别为 1）
            cls_score = cls_score[pos_inds]
            labels = labels[pos_inds]


            # if not hasattr(self, '_debug_printed'):
            #     is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            #     if is_main:
            #         print('\n====== DEBUG (BBoxHeadAVA.loss_and_target) ======')
            #         print('cls_score.shape =', tuple(cls_score.shape))         # 期望 (N, 6)
            #         print('labels.shape    =', tuple(labels.shape))            # 期望 (N, 6)  (单标签 one-hot)
            #         print('row-sum(labels) 前10个 =', labels.float().sum(1)[:10].cpu().tolist())
            #         # 看看是不是每行恰好 1（单标签 one-hot）
            #         probs = cls_score.softmax(1)
            #         top3 = probs.topk(3, dim=1).indices[:10].cpu().tolist()
            #         print('top3 index 前10行 =', top3)
            #         print('===============================================\n')
            #     self._debug_printed = True
        # —— 只打一次：显示本批次真正出现的类别 ID ——  
            if not hasattr(self, '_debug_printed'):
                is_main = (not dist.is_available()) \
                        or (not dist.is_initialized()) \
                        or dist.get_rank() == 0
                if is_main:
                    # 单标签：每行只有一个 1，我们先 argmax 得到类别索引
                    labels_idx = labels.argmax(dim=1)
                    unique_ids = labels_idx.unique().cpu().tolist()
                    print(f"DEBUG batch classes: {unique_ids}")
                self._debug_printed = True
            # ---------- 单标签、无背景 ----------
            # 3) 单标签断言 & idx 转换
            if not self.multilabel:
                assert (labels.sum(1) == 1).all(), '单标签任务要求 one-hot'

                labels_idx = labels.argmax(1)                 # (N,)  ## 将独热编码转换为类别索引
                probs = cls_score.softmax(1)                  # (N,5)  ## 计算每个类别的概率
                pt = probs[torch.arange(labels_idx.numel()), labels_idx] ## 获取每个样本对应真实类别的概率
                ce = F.cross_entropy(cls_score, labels_idx, reduction='none')  ## 计算交叉熵损失

                alpha = self.focal_alpha                     ## 获取Focal Loss的alpha参数
                if isinstance(alpha, torch.Tensor):
                    alpha = alpha[labels_idx]                 # (N,)   # (N,) 为每个样本选择对应的alpha

                focal = alpha * (1 - pt) ** self.focal_gamma * ce    # 计算Focal Loss
                losses['loss_action_cls'] = focal.mean()

                # —— 仅此一次指标计算 ——
                recall_thr, prec_thr, recall_k, prec_k = self.topk_accuracy(
                    cls_score, labels, thr=0.5)
                losses['recall@top1'] = recall_thr
                losses['prec@top1']   = prec_thr
                for i, k in enumerate(self.topk):
                    losses[f'recall@top{k}'] = recall_k[i]
                    losses[f'prec@top{k}']   = prec_k[i]

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

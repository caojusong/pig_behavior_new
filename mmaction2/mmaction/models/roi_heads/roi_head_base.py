# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import bbox2roi
from torch import Tensor

from mmaction.utils import ConfigType, InstanceList, SampleList


class AVARoIHead(StandardRoIHead):

#loss 方法主要用于在目标检测任务里，对检测区域（RoI）进行前向传播并计算损失。
    def loss(self, x: Union[Tensor,
                            Tuple[Tensor]], rpn_results_list: InstanceList,
             data_samples: SampleList, **kwargs) -> dict:
        """即利用上游网络提取的特征,对检测区域(Region of Interest, RoI)进行前向传播，并计算相应的损失。

        Args:
            x:类型可以是 Tensor 或 Tensor 元组，代表上游网络（如卷积神经网络）提取出的图像特征。
            rpn_results_list:一个包含 InstanceData 对象的列表,InstanceData 是自定义的数据结构,这里存储的是区域建议网络(Region Proposal Network, RPN)生成的区域建议，也就是可能包含目标的候选框。
            data_samples:一个包含 ActionDataSample 对象的列表,ActionDataSample 也是自定义的数据结构，代表一批数据样本，里面包含了图像的真实标注信息等。
        Returns:
            方法返回一个字典,字典的键(str 类型)是不同损失项的名称,值(Tensor 类型）是对应损失项的计算结果。例如，可能包含分类损失、边界框回归损失等不同的损失分量。
        """
        assert len(rpn_results_list) == len(data_samples)
        #​​准备 GT 数据​​：
        batch_gt_instances = []
        for data_sample in data_samples:  #从 data_samples 中提取每张图片的真实标注（gt_instances）
            batch_gt_instances.append(data_sample.gt_instances)

        # assign gts and sample proposals
        num_imgs = len(data_samples)    #初始化变量：获取批量图像的数量 num_imgs，并初始化一个空列表 sampling_results，用于存储每张图像的采样结果。
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(rpn_results,                     #分配真实标签：调用 self.bbox_assigner.assign 方法，将当前图像的真实实例 batch_gt_instances[i] 分配给对应的区域建议 rpn_results，得到分配结果 assign_result。
                                                      batch_gt_instances[i],
                                                      None)
            sampling_result = self.bbox_sampler.sample(assign_result,                  #基于分配结果 assign_result，调用 self.bbox_sampler.sample 方法对区域建议进行采样，得到采样结果 sampling_result。采样过程通常会从区域建议中选出正样本（包含目标的建议框）和负样本（不包含目标的建议框）。
                                                       rpn_results,
                                                       batch_gt_instances[i])
            sampling_results.append(sampling_result)                                   #将每张图像的采样结果 sampling_result 添加到 sampling_results 列表中。

        # LFB needs meta_info: 'img_key'
        batch_img_metas = [
            data_samples.metainfo for data_samples in data_samples
        ]

        losses = dict()
        # bbox head forward and loss
        bbox_results = self.bbox_loss(x, sampling_results, batch_img_metas)
        losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x: Union[Tensor, Tuple[Tensor]], rois: Tensor,              
                      batch_img_metas: List[dict], **kwargs) -> dict:

        """Box head forward function used in both training and testing.

        Args:
        #x,主干网络提取的特征图（可能是多尺度特征，如 FPN 的 P2-P5 层）
        #rois	Tensor	形状为 (n, 5) 的 RoI 坐标，每行格式为 [batch_idx, x1, y1, x2, y2]
        #batch_img_metas	List[dict]	每张图片的元信息（如图像尺寸、缩放比例），用于支持可变输入尺寸和增强参数反算

        Returns:
                dict[str, Tensor]: Usually returns a dictionary with keys:

                cls_score (Tensor)：分类得分，其形状通常为 (n, num_classes)，其中 n 是 RoI 的数量,num_classes 是目标的类别数量。该得分表示每个 RoI 属于各个类别的概率。
                bbox_pred (Tensor)：边界框预测值，也称为边界框能量或偏移量。形状通常为 (n, 4 * num_classes)，这里的 4 表示每个边界框需要预测的 4 个参数（如 x, y, w, h 或 x1, y1, x2, y2），每个类别都有对应的一组边界框预测参数。
                bbox_feats (Tensor)：从输入特征中提取的边界框 RoI 特征，用于后续的分类和边界框回归任务。
        """
        #典型实现为 SingleRoIExtractor 或 MultiScaleRoIExtractor（如 FPN）。
        bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)
        #根据 rois 的坐标从特征图 x 中裁剪对应区域。通过 RoI Align/Pooling 将不同大小的区域缩放到固定大小（如 7x7）。
        if self.with_shared_head:
            bbox_feats = self.shared_head(
                bbox_feats,
                feat=global_feat,
                rois=rois,
                img_metas=batch_img_metas)

        cls_score = self.bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Union[Tensor, Tuple[Tensor]],
                  sampling_results: List[SamplingResult],
                  batch_img_metas: List[dict], **kwargs) -> dict:
        """该方法的主要功能是基于上游网络提取的特征,执行边界框头部(bbox head)的前向传播,并计算边界框相关的损失。

        Args:
            x (Tensor or Tuple[Tensor])：上游网络（如卷积神经网络骨干网络，像 ResNet、FPN 等）提取的图像特征。类型可以是单个 Tensor,也可能是多个 Tensor 组成的元组（例如使用 FPN 时会输出不同尺度的特征图元组）。
            sampling_results (List[SamplingResult])：采样结果列表，每个 SamplingResult 对象包含了正负样本的信息,是在之前的步骤中对区域建议(RoIs)进行采样得到的结果。
            batch_img_metas (List[dict])：图像信息列表，列表里每个字典包含对应图像的元信息，这些信息在特征提取、边界框预测以及损失计算等过程中可能会用到。
        Returns:
            cls_score (Tensor)：分类得分，形状一般为 (n, num_classes)，其中 n 是 RoI 的数量,num_classes 是目标的类别数量，该得分表示每个 RoI 属于各个类别的概率。
            bbox_pred (Tensor)：边界框预测值，也称为边界框能量或偏移量。形状通常为 (n, 4 * num_classes)，这里的 4 表示每个边界框需要预测的 4 个参数（如 x, y, w, h 或 x1, y1, x2, y2），每个类别都有对应的一组边界框预测参数。
            bbox_feats (Tensor)：从输入特征中提取的边界框 RoI 特征，用于后续的分类和边界框回归任务。
            loss_bbox (dict)：一个包含边界框损失分量的字典，例如可能包含分类损失、边界框回归损失等不同的损失项。
        """
        #输入​​：sampling_results 是经过筛选的候选框（比如每张图选了256个框）。
        #操作​​：
        #res.priors：候选框的坐标 [x1,y1,x2,y2]（类似快递的“寄件地址”）。
        #bbox2roi：给每个框加上“批次标签”（告诉系统这个框属于哪张图片）。
        #得到的 rois 是形如 [n,5] 的张量，多了一维，第一维是批次标签。  
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois, batch_img_metas)  #预测包裹内容（前向传播）​
        # 计算误差（对比预测和真实）
        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    #在推理阶段（如测试或部署时），对输入图像的特征和RPN生成的候选框进行前向传播，输出最终的检测结果（类别标签和置信度）。
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

        results_list = self.predict_bbox(
            x, batch_img_metas, rpn_results_list, rcnn_test_cfg=self.test_cfg)

        return results_list

    def predict_bbox(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType) -> InstanceList:
    #对每张图片的候选框（RoIs）执行以下操作：
    #提取RoI特征并预测类别得分。
    #后处理（如非极大值抑制NMS）生成最终检测结果
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following
            keys:
                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, batch_img_metas)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        return result_list

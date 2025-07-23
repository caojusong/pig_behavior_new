# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy as np
from mmengine.fileio import exists, list_from_file, load
from mmengine.logging import MMLogger

from mmaction.evaluation import read_labelmap
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class AVADataset(BaseActionDataset):
    """
    STAD dataset for spatial temporal action detection.
    该数据集会加载原始视频帧/视频文件、边界框、提案(proposals)并应用指定的转换操作，最终返回一个包含帧张量和其他信息的字典。

    此数据集可以从以下文件中加载信息：

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    特别地，`proposal_file` 是一个 pickle 文件，其中包含 ``img_key``（格式为 ``{video_id},{timestamp}``）。下面是一个 pickle 文件的示例：

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    参数:
        ann_file (str): 标注文件的路径，例如 ``ava_{train, val}_{v2.1, v2.2}.csv``。
        exclude_file (str): 排除时间戳文件的路径，例如 ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``。
        pipeline (List[Union[dict, ConfigDict, Callable]]): 一系列的数据转换操作。
        label_file (str): 标签文件的路径，例如 ``ava_action_list_{v2.1, v2.2}.pbtxt`` 或 
                          ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``。默认值为 None。
        filename_tmpl (str): 每个文件名的模板。默认值为 'img_{:05}.jpg'。
        start_index (int): 考虑到不同的文件名格式，为帧指定起始索引。对于 AVA 数据集，帧索引从 1 开始，因此该值应设为 1。默认值为 1。
        proposal_file (str): 提案文件的路径，例如 ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``。默认值为 None。
        person_det_score_thr (float): 人物检测分数的阈值，分数高于该阈值的边界框将被使用。
            请注意,0 <= person_det_score_thr <= 1。如果没有提案的检测分数高于该阈值,则会使用检测分数最高的那个。默认值:0.9。
        num_classes (int): 数据集的类别数量。默认值:81。(AVA 有 80 个动作类别，额外增加 1 维用于潜在用途）
        custom_classes (List[int], 可选): 原始数据集中类 ID 的子集。请注意，不应选择 0,并且 ``num_classes`` 应等于 ``len(custom_classes) + 1``。
        data_prefix (dict 或 ConfigDict): 存储视频帧的目录路径。默认值为 ``dict(img='')``。
        test_mode (bool): 构建测试或验证数据集时设置为 True。默认值为 False。
        modality (str): 数据的模态。支持 ``RGB``、``Flow``。默认值为 ``RGB``。
        num_max_proposals (int): 要存储的最大提案数量。默认值为 1000。
        timestamp_start (int): 包含的时间戳的起始点。默认值参考官方网站。默认值为 902。
        timestamp_end (int): 包含的时间戳的结束点。默认值参考官方网站。默认值为 1798。
        use_frames (bool): 是否使用原始帧作为输入。默认值为 True。
        fps (int): 覆盖数据集的默认帧率。如果设置为 1,则表示按帧计算时间戳,例如 MultiSports 数据集。否则按秒计算。默认值为 30。
        multilabel (bool): 确定是否为多标签识别任务。默认值为 True。
    """

    def __init__(self,
                 ann_file: str,                                 #标注文件路径，格式如 ava_{train, val}_{v2.1, v2.2}.csv
                 pipeline: List[Union[ConfigType, Callable]],   #数据转换序列，包含一系列数据处理操作。
                 exclude_file: Optional[str] = None,
                 label_file: Optional[str] = None,               #可选参数，标签文件路径，默认为 None。
                 filename_tmpl: str = 'img_{:05}.jpg',           #文件名模板，默认为 'img_{:05}.jpg'。
                 start_index: int = 1,                           #起始索引，默认为 1。  
                 proposal_file: str = None,                      #可选参数，建议文件路径，默认为 None。
                 person_det_score_thr: float = 0.9,
                 num_classes: int = 6,
                 custom_classes: Optional[List[int]] = None,
                 data_prefix: ConfigType = dict(img=''),          #数据前缀，默认为 dict(img='')。
                 modality: str = 'RGB',                           #数据模态，默认为 'RGB'。
                 test_mode: bool = False,
                 num_max_proposals: int = 1000,
                 timestamp_start: int = 0,
                 timestamp_end: int = 300,
                 use_frames: bool = True,                         #是否使用原始帧作为输入，默认为 True。
                 fps: int = 30,
                 multilabel: bool = True,                         #是否为多标签识别任务，默认为 True。
                 **kwargs) -> None: 
        self._FPS = fps  # Keep this as standard                  #存储帧率。
        self.custom_classes = custom_classes                      
        if custom_classes is not None:                            # 确保自定义类别数量加上背景类（索引 0）等于总类别数
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = list([0] + custom_classes)
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.num_max_proposals = num_max_proposals
        self.filename_tmpl = filename_tmpl
        self.use_frames = use_frames                              # 存储是否使用原始帧的标志
        self.multilabel = multilabel                              # 存储是否为多标签任务的标志

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

        if self.proposal_file is not None:
            self.proposals = load(self.proposal_file)
        else:
            self.proposals = None
    def parse_img_record(self, img_records: List[dict]) -> tuple:
        #合并同一时间点同一实体的多行标注记录​​。在AVA数据集中，同一个人的边界框（bbox）可能对应多个动作标签（如"走路"和"挥手"同时发生），此方法将这些分散的记录合并为一条结构化数据。
        """Merge image records of the same entity at the same time.

        Args:
            img_records (List[dict]): List of img_records (lines in AVA
                annotations).

        Returns:
            Tuple(list): A tuple consists of lists of bboxes, action labels and
                entity_ids.
        """
        #函数返回一个元组，包含边界框列表、动作标签列表和实体 ID 列表。
        bboxes, labels, entity_ids = [], [], []  #初始化三个空列表 bboxes、labels 和 entity_ids，用于存储合并后的边界框、动作标签和实体 ID。
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)

            selected_records = [
                x for x in img_records
                if np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            num_selected_records = len(selected_records)
            img_records = [
                x for x in img_records if
                not np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            if self.multilabel:
                label = np.zeros(self.num_classes, dtype=np.float32)
                label[valid_labels] = 1.
            else:
                label = valid_labels

            labels.append(label)
            entity_ids.append(img_record['entity_id'])
        bboxes = np.stack(bboxes)   # 会将 bboxes 列表中的元素沿着新的轴堆叠起来，形成一个 NumPy 数组。
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def load_data_list(self) -> List[dict]:
        #该方法的作用是加载 AVA 数据集的标注信息，并将其整理成特定格式的列表返回。
        """Load AVA annotations."""
        exists(self.ann_file)        # 检查标注文件是否存在
        data_list = []               # 用于存储最终的视频信息列表
        records_dict_by_img = defaultdict(list)    # 以图片键为键，存储对应视频信息列表的字典，默认值为空列表
        fin = list_from_file(self.ann_file)        # 逐行读取标注文件内容
        for line in fin:                           # 遍历标注文件的每一行
            line_split = line.strip().split(',')   # 去除行首尾空白字符并按逗号分割
            label = int(line_split[6])             # 提取标注类别，位于分割后列表的第 7 个元素
            if self.custom_classes is not None: 
                if label not in self.custom_classes:
                    continue
                label = self.custom_classes.index(label)

            video_id = line_split[0]                # 提取视频 ID，位于分割后列表的第 1 个元素
            timestamp = int(line_split[1])  # count by second or frame.  # 提取时间戳，位于分割后列表的第 2 个元素，按秒或帧计数
            img_key = f'{video_id},{timestamp:04d}' # 生成图片键，格式为 "视频 ID, 四位数时间戳"

            entity_box = np.array(list(map(float, line_split[2:6])))   # 提取实体边界框信息，位于分割后列表的第 3 到 6 个元素，并转换为浮点数数组
            entity_id = int(line_split[7])           # 提取实体 ID，位于分割后列表的第 8 个元素
            if self.use_frames:                       # 如果使用原始帧作为输入
                shot_info = (0, (self.timestamp_end - self.timestamp_start) *        # 计算镜头信息，起始帧为 0，结束帧为 (时间戳结束值 - 时间戳起始值) * 帧率
                             self._FPS)
            # for video data, automatically get shot info when decoding
            else:
                shot_info = None

            video_info = dict(
                video_id=video_id,
                timestamp=timestamp,
                entity_box=entity_box,
                label=label,
                entity_id=entity_id,
                shot_info=shot_info)
            records_dict_by_img[img_key].append(video_info)

        for img_key in records_dict_by_img:     #遍历 records_dict_by_img 字典，对每个图片键对应的数据进行处理：
            video_id, timestamp = img_key.split(',')
            bboxes, labels, entity_ids = self.parse_img_record(     #调用 self.parse_img_record 方法合并同一图片键下的记录，得到边界框、标签和实体 ID 列表。
                records_dict_by_img[img_key])   
            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids)  #构建标注信息字典 ann
            #构建文件夹路径
            frame_dir = video_id   #基础路径：video_id 作为文件夹名称
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)     #根据 self.data_prefix['img'] 的值更新 frame_dir  # 拼接数据前缀，例如 data_prefix.img = "ava_frames/"
            # 最终 frame_dir = "ava_frames/053oq2sB3oU/"
            video_info = dict( 
                frame_dir=frame_dir,       # 帧所在的文件夹路径
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,          #img_key = f'{video_id},{timestamp:04d}'
                shot_info=shot_info,
                fps=self._FPS,
                ann=ann)
            if not self.use_frames:
                video_info['filename'] = video_info.pop('frame_dir')  #构建最终的视频信息字典 video_info，若不使用原始帧，将 frame_dir 键改为 filename。
            data_list.append(video_info)

        return data_list

    def filter_data(self) -> List[dict]:
        #过滤掉数据列表中不符合要求的记录，返回有效的数据列表
        """Filter out records in the exclude_file."""
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.data_list)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, data_info in enumerate(self.data_list):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (data_info['video_id'] == video_id
                            and data_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break

        logger = MMLogger.get_current_instance()
        logger.info(f'{len(valid_indexes)} out of {len(self.data_list)}'
                    f' frames are valid.')
        data_list = [self.data_list[i] for i in valid_indexes]

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """
        根据索引获取标注信息。

        Args:
            idx (int): 数据列表中的索引。

        Returns:
            dict: 包含数据信息的字典。
        """
        data_info = super().get_data_info(idx)
        img_key = data_info['img_key']  #从数据信息中提取图片键

        data_info['filename_tmpl'] = self.filename_tmpl       # 将文件名模板添加到数据信息中
        data_info['timestamp_start'] = self.timestamp_start   # # 将时间戳起始值添加到数据信息中
        data_info['timestamp_end'] = self.timestamp_end

        if self.proposals is not None:                                  #如果存在提案文件
            if img_key not in self.proposals:                           # 若图片键不在提案字典中
                data_info['proposals'] = np.array([[0, 0, 1, 1]])       # 设置默认提案框
                data_info['scores'] = np.array([1])                     #设置默认检测分数
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals[:, :4]
                    data_info['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals

                assert data_info['proposals'].max() <= 1 and \
                    data_info['proposals'].min() >= 0, \
                    (f'relative proposals invalid: max value '
                     f'{data_info["proposals"].max()}, min value '
                     f'{data_info["proposals"].min()}')

        ann = data_info.pop('ann')
        data_info['gt_bboxes'] = ann['gt_bboxes']
        data_info['gt_labels'] = ann['gt_labels']
        data_info['entity_ids'] = ann['entity_ids']

        return data_info
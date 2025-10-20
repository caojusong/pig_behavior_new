####，用的transformer进行的融合
custom_imports = dict(
    imports=[
        'mmaction.datasets.transforms.print_inds',
        'mmaction.datasets.transforms.LoadTrackInfo',   # ✅ 保持与注册名一致
        'mmaction.models.roi_heads.track_modules',
        'mmaction.models.roi_heads.scm_fusion',
        'mmaction.models.roi_heads.roi_head',           # ✅ 你的 AVARoIHead 在这里
        'mmaction.models.roi_heads.cross_attention',
    ],
    allow_failed_imports=False
)
_base_ = '../../_base_/default_runtime.py'

url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        # ===================== 关键改动（与 K 对齐）=====================
        # 你的新版 LoadTrackInfo 输出 (M,T, 2+1+K)，若 K=9 -> input_dim=12
        track_cfg=dict(
            input_dim=12,          # <-- ★ 改成 2+1+K
            hidden_dim=1024,
            output_dim=2304,
            dropout_ratio=0.3
        ),
        # =====================这个是没加senet的配置文件=========================================
        fusion_cfg=dict(
            vis_dim=2304, traj_dim=2304, model_dim=256,
            num_layers=2, num_heads=8, dropout=0.3,
            output_dim=2304, enhanced=True,
            fusion_type='mean'
        ),
        # =====================这个是加senet的配置文件=========================================
        fusion_cfg=dict(
            vis_dim=2304, traj_dim=2304,
            model_dim=256, num_layers=2, num_heads=8,
            dropout=0.3, output_dim=2304,
            fusion_type='mean',         # 先 mean 更稳，必要时再试 'transformer'
            # —— ① 分支端 SE（开启）——
            use_ca_vis=True,
            use_ca_traj=True,
            ca_reduction=16,
            # —— ② 融合后 SE（先关，稳了再开）——
            use_ca_post=False,
            # mean 融合下轨迹注入比例（0.25/0.33/0.5 可扫）
            gate_alpha=0.25
        )
        ####################################################################################
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            topk=(1, 3, 5),
            background_class=True,
            in_channels=2304,
            num_classes=6,
            multilabel=False,
            dropout_ratio=0.5,
            focal_gamma=0,
            focal_alpha=1,
        )),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
data_root = '/home/caojs/mmaction2/data03/ava/rawframes/'
anno_root = '/home/caojs/mmaction2/data03/ava/annotations/'

ann_file_train = f'{anno_root}/ava_train_v2.1.csv'
ann_file_val = f'{anno_root}/ava_val_v2.1.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'

label_file = f'{anno_root}/ava_action_list_v2.1.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.pkl'

file_client_args = dict(io_backend='disk')

# ===================== 关键改动：LoadTrackInfo 参数 =====================
# - origin_alpha: 原点沿“中心→饮水器”方向平移比例（0=不平移；0.5=中心与水器中点；1=水器）
# - max_neighbors: K（最近邻数）。若改这里，也要同步改上面的 input_dim=2+1+K
# - water_center_norm: 饮水器在图像归一化坐标中的位置
# =====================================================================
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode', **file_client_args),

    dict(
        type='LoadTrackInfo',
        track_base_path='/home/caojs/mmaction2/data03/ava/trackID',
        iou_threshold=0.30,
        follow_iou_ratio=0.15,
        verbose=False,
        # ===== 新增 / 与新版实现对齐 =====
        max_neighbors=9,                         # K
        origin_alpha=0.5,                        # 原点平移比例（0~1）
        water_center_norm=(0.0056745, 0.4214915) # 饮水器坐标（按你的默认）
    ),

    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', algorithm_keys=['track_vector']),
]

val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),

    dict(
        type='LoadTrackInfo',
        track_base_path='/home/caojs/mmaction2/data03/ava/trackID',
        iou_threshold=0.30,
        follow_iou_ratio=0.15,
        verbose=False,
        # ===== 与训练一致 =====
        max_neighbors=9,
        origin_alpha=0.5,
        water_center_norm=(0.0056745, 0.4214915)
    ),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', algorithm_keys=['track_vector']),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
        start_index=1,
))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True,
        start_index=1,
))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val,
    num_classes=6)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.2, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[10, 15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=1e-6),
    clip_grad=dict(max_norm=40, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=128)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)









######这个为引入Cross-Attn的配置文件，两个配置文件是大差不差的######
custom_imports = dict(
    imports=[
        'mmaction.datasets.transforms.print_inds',
        'mmaction.datasets.transforms.LoadTrackInfo',   # ✅ 保留
        'mmaction.models.roi_heads.scm_fusion_xattn',   # ✅ 新增：Cross-Attn 模块
        'mmaction.models.roi_heads.roi_head',           # ✅ 你的 AVARoIHead
    ],
    allow_failed_imports=False
)
_base_ = '../../_base_/default_runtime.py'

url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',

        # ======= 新的 fusion_cfg（Cross-Attn）=======
        fusion_cfg=dict(
            vis_dim=2304,
            traj_dim=3,        # 2 + 1 + K（K=9 → 12）
            embed_dim=256,
            num_heads=4,
            max_seq_len=64,     # ≥ 轨迹长度 T 的上界（比如 T=32）
            dropout=0.3,
            output_dim=2304,    # 要与 bbox_head.in_channels 一致
            traj_scale=0.3      # 注入强度：0.3~0.6 起步更稳
        ),
        # ===========================================

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            topk=(1, 3, 5),
            background_class=True,
            in_channels=2304,
            num_classes=6,
            multilabel=False,
            dropout_ratio=0.5,
            focal_gamma=0,
            focal_alpha=1,
        )),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
data_root = '/home/caojs/mmaction2/data03/ava/rawframes/'
anno_root = '/home/caojs/mmaction2/data03/ava/annotations/'

ann_file_train = f'{anno_root}/ava_train_v2.1.csv'
ann_file_val = f'{anno_root}/ava_val_v2.1.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'

label_file = f'{anno_root}/ava_action_list_v2.1.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.pkl'

file_client_args = dict(io_backend='disk')

# ========== LoadTrackInfo 参数（与 XAttn 对齐）==========
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=False, debug=False, jitter=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(
        type='LoadTrackInfo',
        track_base_path='/home/caojs/mmaction2/data03/ava/trackID',
        iou_threshold=0.30,
        follow_iou_ratio=0.15,
        verbose=False,
        max_neighbors=0,                         # K
        origin_alpha=0,                        # 原点平移比例（0~1）
        water_center_norm=(0.0056745, 0.4214915)
    ),

    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # 如果 LoadTrackInfo 返回了 mask，就一起打包；否则换成 ['track_vector']
    dict(type='PackActionInputs', algorithm_keys=['track_vector']),
]

val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True, debug=False),
    dict(type='RawFrameDecode', **file_client_args),

    dict(
        type='LoadTrackInfo',
        track_base_path='/home/caojs/mmaction2/data03/ava/trackID',
        iou_threshold=0.30,
        follow_iou_ratio=0.15,
        verbose=False,
        max_neighbors=0,
        origin_alpha=0,
        water_center_norm=(0.0056745, 0.4214915)
    ),

    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', algorithm_keys=['track_vector']),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
        start_index=1,
))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True,
        start_index=1,
))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val,
    num_classes=6)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.2, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[10, 15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-6),
    paramwise_cfg=dict(
        custom_keys={
            'roi_head.scm_fusion': dict(lr_mult=0.33),  # ← XAttn 降学习率
        }
    ),
    clip_grad=dict(max_norm=40, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=128)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

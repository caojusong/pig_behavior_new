custom_imports = dict(
    imports=[
        # 下面这几行一定要加入，路径和文件名都要写全
        'mmengine.hooks.early_stopping_hook',
    ],
    allow_failed_imports=False
)
_base_ = '../../_base_/default_runtime.py'

url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')
# 在这里直接定义 default_hooks
# 覆盖默认 hook
# 追加到 config 最后即可

#custom_hooks = [
#    dict(
#        type='EarlyStoppingHook',
#        monitor='mAP/overall',
#        rule='greater',
#        patience=50,          # 你的早停容忍
#        priority='VERY_LOW'   # 一定要低于 LoggerHook 的 NORMAL
#    )
#]



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
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=False,
            in_channels=2304,
            num_classes=6,
            multilabel=False,                 #使用单标签
            dropout_ratio=0.5,
            #focal_gamma=2.0,    # Focal Loss γ，一般设 1~2
            #focal_alpha=0.25,   # Focal Loss α，常用 0.25
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
train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),     #/home/caojs/mmaction2/mmaction/datasets/transforms/loading.py
    #从视频帧序列里随机采样出一段连续的帧。clip_len=32 表明采样的帧数为 32 帧；frame_interval=2 意味着每隔 2 帧采样一次。此操作定义于 /home/caojs/mmaction2/mmaction/datasets/transforms/loading.py 文件。
    dict(type='RawFrameDecode', **file_client_args),
    #对采样得到的原始视频帧进行解码操作，将其转换为可处理的图像数据。**file_client_args 是将 file_client_args 字典里的键值对作为参数传入，用于指定文件读取的后端（如 io_backend='disk' 表示从磁盘读取）。
    dict(type='RandomRescale', scale_range=(256, 320)),
    #对解码后的图像进行随机缩放。scale_range=(256, 320) 表示图像的短边会被随机缩放到 256 到 320 像素之间，同时保持图像的宽高比。
    dict(type='RandomCrop', size=256),
    #对缩放后的图像进行随机裁剪，裁剪出一个大小为 256x256 像素的图像块。
    dict(type='Flip', flip_ratio=0.5),
    #以 0.5 的概率对裁剪后的图像进行水平翻转，这是一种数据增强手段，可增加训练数据的多样性。
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    #对处理后的图像数据进行格式调整。input_format='NCTHW' 表示将数据格式调整为 (batch_size, channels, time, height, width)；collapse=True 意味着会对时间维度进行折叠处理。
    dict(type='PackActionInputs') #将处理好的数据打包成适合模型输入的格式
]



# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
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
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val)
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
        # 关键：对 backbone 参数乘 lr_mult = 0 ⇒ 只训练 ROI Head 
    #paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.)}),    #现在通过 lr_mult=0. 冻住权重最稳       
    clip_grad=dict(max_norm=40, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)

#把 backbone 的 lr_mult 设为 0.，等价于：学习率 (lr) × 0 = 0，，训练过程中优化器对 backbone 参数的 梯度更新被完全抑制 ，换句话说，backbone 被冻结（freeze），只对其余部分（RoI Head / 分类头等）进行训练或微调


default_scope = 'mmaction'  # 默认注册表范围，用于查找模块。参考 https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'), # 将运行时信息更新到消息中心的钩子
    timer=dict(type='IterTimerHook'),# 用于记录迭代过程中花费的时间的日志记录器
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),# 打印日志的间隔,# 忽略每个 epoch 中最后几次迭代的日志
    param_scheduler=dict(type='ParamSchedulerHook'),# 更新优化器中的某些超参数的钩子
    checkpoint=dict(type='CheckpointHook', # 定期保存权重的钩子
                    interval=10, # 保存周期
                    save_best='auto',# 在评估过程中测量最佳权重的指标
                    max_keep_ckpts=1),# 保留的最大权重文件数量
    sampler_seed=dict(type='DistSamplerSeedHook'),# 用于分布式训练的数据加载采样器
    sync_buffers=dict(type='SyncBuffersHook'))#在每个 epoch 结束时同步模型缓冲区的钩子

env_cfg = dict(
    cudnn_benchmark=False,  #  # 是否启用 cudnn 的基准测试
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),# 设置多进程的参数
    dist_cfg=dict(backend='nccl'))# 设置分布式环境的参数，也可以设置端口

log_processor = dict(type='LogProcessor', # 用于格式化日志信息的日志处理器
                     window_size=20, # 默认平滑间隔
                     by_epoch=True) # 是否使用 epoch 类型格式化日志

vis_backends = [dict(type='LocalVisBackend')]# 可视化后端的列表
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends) # 可视化器的名称

log_level = 'INFO'# 日志级别
load_from = None
resume = False

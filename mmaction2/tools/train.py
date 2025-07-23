# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS


def parse_args():
     # 创建一个 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description='Train a action recognizer')
     # 添加一个位置参数，用于指定训练配置文件的路径
    parser.add_argument('config', help='train config file path')
     # 添加一个可选参数，用于指定保存日志和模型的目录
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # 添加一个可选参数，用于指定是否从检查点恢复训练
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    # 添加一个可选参数，用于启用自动混合精度训练
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    # 添加一个可选参数，用于指定在训练过程中是否不评估检查点
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    # 添加一个可选参数，用于指定是否根据实际批量大小和原始批量大小自动缩放学习率
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    # 添加一个可选参数，用于指定随机种子
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # 添加一个可选参数，用于指定是否为不同的进程设置不同的随机种子
    parser.add_argument(
        '--diff-rank-seed',
        action='store_true',
        help='whether or not set different seeds for different ranks')
    # 添加一个可选参数，用于指定是否为 CUDA 后端设置确定性选项
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # 添加一个可选参数，用于覆盖配置文件中的某些设置
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # 添加一个可选参数，用于指定作业启动器
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
     # 添加一个可选参数，用于指定本地进程的排名
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # 解析命令行参数
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """
    将命令行参数合并到配置文件中。

    Args:
        cfg (Config): 配置文件对象。
        args (argparse.Namespace): 命令行参数对象。

    Returns:
        Config: 合并后的配置文件对象。
    """

    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set random seeds
    if cfg.get('randomness', None) is None:
        cfg.randomness = dict(
            seed=args.seed,
            diff_rank_seed=args.diff_rank_seed,
            deterministic=args.deterministic)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    """
    主函数，用于执行动作识别器的训练流程。

    此函数负责解析命令行参数，加载配置文件，合并参数到配置中，
    构建训练运行器，并最终启动训练过程。
    """
    #调用 parse_args 函数，该函数会解析命令行输入的参数，并将解析结果存储在 args 对象中。
    args = parse_args()
    # 从指定的配置文件路径加载配置信息，并将其存储在 cfg 对象中
    cfg = Config.fromfile(args.config)
    # 打印出配置文件的详细内容，方便用户确认配置信息
    print(cfg.pretty_text)
    # merge cli arguments to config
     # 将命令行参数合并到配置文件中
    cfg = merge_args(cfg, args)

    # build the runner from config
     # 构建训练运行器
    # 检查配置文件中是否包含 'runner_type' 字段
    if 'runner_type' not in cfg:
        # build the default runner
        # 检查配置文件中是否包含 'runner_type' 字段
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        # 如果包含，则从 RUNNERS 注册表中根据配置文件构建一个自定义的运行器对象
        runner = RUNNERS.build(cfg)
    # 调用运行器对象的 train 方法，开始执行训练过程
    # start training
    model = runner.model
    print(model)
    runner.train()


if __name__ == '__main__':
    main()
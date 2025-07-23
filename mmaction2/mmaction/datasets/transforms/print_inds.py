# mmaction/datasets/transforms/print_inds.py
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PrintFirstSample(BaseTransform):
    """只打印第一个 sample 的 frame_inds，训练时只执行一次。"""

    def transform(self, results: dict) -> dict:
        # worker_id 为 0 且 idx == 0 时打印一次
        if results.get('sample_idx', 0) == 0 and results.get('worker_id', 0) == 0:
            inds = results['frame_inds']
            center = inds[len(inds) // 2]
            print('=' * 60)
            print(f"[PrintFirstSample] frame_inds = {inds.tolist()}")
            print(f"[PrintFirstSample] center    = {center}")
            print('=' * 60, flush=True)
        return results

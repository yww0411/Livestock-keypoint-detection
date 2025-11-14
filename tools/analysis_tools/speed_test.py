# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmengine.config import DictAction

from mmpose.apis.inference import init_model

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get inference speed information from a model config')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--device', default='cpu', help='Device used for model initialization')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=4,
        help='Input batch size. If specified and greater than 1, it takes a '
             'callable method that generates a batch input. Otherwise, it will '
             'generate a random tensor with input shape to calculate FLOPs.')
    parser.add_argument(
        '--cycles',
        '-c',
        type=int,
        default=50,
        help='Number of cycles, cycle n times and then take the average')
    args = parser.parse_args()
    return args


def batch_constructor(flops_model, batch_size, input_shape):
    """Generate a batch of tensors to the model."""
    batch = {}
    inputs = torch.randn(batch_size, *input_shape).new_empty(
        (batch_size, *input_shape),
        dtype=next(flops_model.parameters()).dtype,
        device=next(flops_model.parameters()).device)
    batch['inputs'] = inputs
    return batch


def inference(args, input_shape):
    model = init_model(
        args.config,
        checkpoint=None,
        device=args.device,
        cfg_options=args.cfg_options)

    if hasattr(model, '_forward'):
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    batch = batch_constructor(model, args.batch_size, input_shape)
    model.eval()
    # warm up 显卡
    with torch.no_grad():
        for i in range(5):
            _ = model(**batch)

        total_time = 0.0
        num_iterations = 0
        cpu_total_time = 0.0
        # 正式开始计算
        for i in range(100):
            torch.cuda.synchronize()  # 等待GPU操作完成
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()  # 记录开始时间
            _ = model(**batch)  # 进行推理
            end_time.record()  # 记录结束时间
            torch.cuda.synchronize()  # 等待GPU操作完成
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 计算推理时间（秒）

            total_time += elapsed_time
            num_iterations += args.batch_size

    average_time = total_time / num_iterations
    fps = 1.0 / average_time
    print(f'device:{args.device}')
    print(f"Average inference time: {average_time:.6f} s")
    print(f"Average FPS: {fps:.2f}")


def main():
    args = parse_args()

    if len(args.input_shape) == 1:
        input_shape = (3, args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = (3,) + tuple(args.input_shape)
    else:
        raise ValueError('invalid input shape')

    if 'cuda' in args.device:
        assert torch.cuda.is_available(
        ), 'No valid cuda device detected, please double check...'
    inference(args, input_shape)
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()


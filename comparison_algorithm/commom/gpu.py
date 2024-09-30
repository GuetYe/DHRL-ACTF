# -*- coding: utf-8 -*-
"""
@File     : gpu.py
@Date     : 2022-09-07 15:20
@Author   : Terry_Li  - 既然选择了远方，便只顾风雨兼程。
IDE       : PyCharm
@Mail     : 1419727833@qq.com
"""
import torch


def gpu():
    """
    GPU加速指令
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检查是否可用GPU进行加速
    return device

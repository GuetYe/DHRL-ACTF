# -*- coding: utf-8 -*-
"""
@File     : network_shorest_path.py
@Date     : 2022-11-14
@Author   : Terry_Li  -- 人生就是在不断结交新的朋友。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
from pathlib import Path
import os
import pickle

def create_weight_pickle_path():
    # 创建一个用来存放pickle文件的路径
    WORK_DIR = Path.cwd().parent
    pickle_path = WORK_DIR / "ryu/weight_pickle"
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)


def convert_file():
    """
    将文本文件转成pickle文件
    """
    for name in os.listdir(setting.TXT_PATH):
        # Read as bytes
        with open("{}/{}".format(setting.TXT_PATH, name), 'rb') as ft:
            data = ft.read()
        # Save as pickle
        create_weight_pickle_path()
        with open('{}/{}pickle'.format(setting.PICKLE_PATH, name.split("txt")[0]), 'wb') as fp:
            pickle.dump(data, fp)


if __name__ == "__main__":
    convert_file()

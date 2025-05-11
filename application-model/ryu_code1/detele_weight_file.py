# -*- coding: utf-8 -*-
"""
@File     : detele_weight_file.py
@Date     : 2022-07-26
@Author   : Terry_Li  --这个世界，从来都需要你独当一面。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import shutil
import os
import setting


def del_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        pass


if __name__ == "__main__":
    txt_path = setting.TXT_PATH  # 文件路径
    pickle_path = setting.PICKLE_PATH 
    del_file(txt_path)
    del_file(pickle_path)

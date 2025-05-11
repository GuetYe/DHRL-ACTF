# -*- coding: utf-8 -*-
"""
@File     : network_delay.py
@Date     : 2022-07-20
@Author   : Terry_Li  -- 在文字还未成为代码的时代，这里的人们靠程序执行侠义。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
# parameter settings
from pathlib import Path

FIRST_FLAG = True

SCHEDULE_PERIOD = 8

DISCOVERY_PERIOD = 5

MONITOR_PERIOD = 5  # monitor period, bw

PRINT_SHOW = True  # True or Flase print the relust

METHOD = 'dijkstra'  # the calculation method of shortest path

FACTOR = 0.9  # the coefficient of 'bw' , 1 - FACTOR is the coefficient of 'delay'

DELAY_PERIOD = 3  # detector period, delay

WEIGHT = 'bw'

WORK_DIR = Path.cwd().parent


LINKS_INFO =  WORK_DIR / "mininet-wifi/links_info/links_info3.xml"  # 链路信息的xml文件路径



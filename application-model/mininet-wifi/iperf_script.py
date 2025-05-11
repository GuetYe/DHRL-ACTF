# -*- coding: utf-8 -*-
"""
@File     : network_shorest_path.py
@Date     : 2022-10-28
@Author   : Terry_Li  -- 长风破浪会有时，直挂云帆济沧海。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import argparse
import shutil
from pathlib import Path

import numpy as np


def read_npy(file=None):
    """
    读取npy类型的数据
    """
    if file is None:
        file = args.file  # 读取文件的
    tms = np.load(file)
    return tms


def create_script(tms):
    label = 0
    tms = np.transpose(tms, (2, 0, 1))  # 更改数据
    for tm in tms:
        # FOR CREATING FOLDERS PER TRAFFIC MATRIX
        Path(f'./iperfTM/TM-{label}').mkdir(parents=True, exist_ok=True)  # 创建父级文件夹
        nameTM = Path(f'./iperfTM/TM-{label}')
        label += 1  # label标签自加
        print('-----', nameTM)
        Path.mkdir(nameTM, exist_ok=True)  # 创建子级文件夹

        # -------------------FLOWS----------------------------------------
        # FOR CREATING FOLDERS PER NODE
        for i in range(len(tm[0])):
            Path.mkdir(nameTM / Path('Clients'), exist_ok=True)  # 在子目录下创建客户端指令文件
            Path.mkdir(nameTM / Path('Servers'), exist_ok=True)  # 在子目录下创建服务器指令文件

        # Default parameters
        time_duration = args.time_duration
        port = args.port
        ip_dest = args.ip_dest
        throughput = args.throughput  # take it in kbps from TM

        # UDP with time = 10s
        #  -c: ip_destination
        #  -b: throughput in k,m or g (Kbps, Mbps or Gbps)
        #  -t: time in seconds

        # SERVER SIDE
        # iperf3 -s

        # CLIENT SIDE with iperf3
        # iperf3 -c <ip_dest> -u -p <port> -b <throughput> -t <duration> -V -J

        # As we do not consider throughput in the same node, when src=dest the thro = 0
        for src in range(len([tm[0]])):
            for dst in range(len(tm[0])):
                if src == dst:
                    print("src:", src, "dst:", dst)
                    tm[src][dst] = 0.0

        for src in range(1, len(tm[0]) + 1):
            with open(str(nameTM) + "/Clients/client_{0}.txt".format(str(src)), 'w') as fileClient:
                # outputstring_a1 = "#!/bin/bash \necho Generating traffic..."
                # fileClient.write(outputstring_a1)  # 往文件里面输入文本
                for dst in range(1, len(tm[0]) + 1):
                    throughput = float(tm[src - 1][dst - 1])
                    # throughput_g = throughput /(100) # scale the throughput value to mininet link capacities
                    temp1 = ''
                    if src != dst:
                        temp1 = ''
                        temp1 += '\n'
                        temp1 += 'iperf3 -c'
                        temp1 += ' 192.168.0.{0}'.format(str(dst))
                        if dst > 9:
                            temp1 += ' -p {0}0{1}'.format(str(src), str(dst))
                        else:
                            temp1 += ' -p {0}00{1}'.format(str(src), str(dst))
                        temp1 += ' -u -b ' + str(format(throughput, '.2f')) + 'M'  # 发流指令
                        # temp1 += '-w 256k -t ' + str(time_duration)
                        temp1 += ' -t ' + str(time_duration)
                        temp1 += ' >/dev/null 2>&1 &\n'  # at the end of the line it's for running the process in bkg
                        temp1 += 'sleep 0.4'
                    fileClient.write(temp1)

        for dst in range(len(tm[0])):
            dst_ = dst + 1
            with open(str(nameTM) + "/Servers/server_{0}.txt".format(str(dst_)), 'w') as fileServer:
                # outputstring_a2 = "#!/bin/bash \necho Initializing server listening...."
                # fileServer.write(outputstring_a2)  # 往文件里面输入文本
                for src in range(len(tm[0])):
                    src_ = src + 1
                    temp2 = ''
                    if src != dst:
                        temp2 = ''
                        temp2 += '\n'
                        temp2 += 'iperf3 -s'
                        temp2 += ''
                        if dst_ > 9:
                            temp2 += ' -p {0}0{1}'.format(str(src_), str(dst_))
                        else:
                            temp2 += ' -p {0}00{1}'.format(str(src_), str(dst_))
                        temp2 += ' -1'
                        temp2 += ' >/dev/null 2>&1 &\n'  # at the end of the line it's for running the process in bkg
                        temp2 += 'sleep 0.3'
                    fileServer.write(temp2)
                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate traffic matrices")
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--time_duration", default=20, help="time_duration")
    parser.add_argument("--port", default=2022, help="port")
    parser.add_argument("--ip_dest", default="192.168.0.1", help="ip_dest")
    parser.add_argument("--throughput", default=0.0, help="take it in kbps from TM")
    parser.add_argument("--file", default=r'tm_statistic/communicate_tm.npy', help="take it in kbps from TM")

    args = parser.parse_args()
    shutil.rmtree("iperfTM") # 删除iperfTM的文件
    tms = read_npy()  # 读取npy文件
    create_script(tms)

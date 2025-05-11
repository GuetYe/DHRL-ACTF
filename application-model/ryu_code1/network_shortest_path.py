  # -*- coding: utf-8 -*-
"""
@File     : network_shorest_path.py
@Date     : 2022-07-24
@Author   : Terry_Li  -- where there is a will, there is a  way。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
from pathlib import Path

import networkx as nx
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER
from ryu.lib import hub
from ryu.lib.packet import packet, arp, ipv4, ethernet
from ryu.ofproto import ofproto_v1_3

import setting
import os
import time


class NetworkShortestPath(app_manager.RyuApp):
    """测量链路的最短路径"""

    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NetworkShortestPath, self).__init__(*args, **kwargs)
        self.name = "network_shorest_path"  # 本地模块的名字，方便其他模块的调用
        self.count = 1  # 设置文件的序号
        self.now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # 获取当前的时间
        self.network_structure = lookup_service_brick("discovery")  # 调用拓扑的模块
        self.network_monitor = lookup_service_brick("monitor")  # 调用监控的模块
        self.network_delay = lookup_service_brick("detector")  # 调用测量时延的模块

        self.shortest_thread = hub.spawn(self.super_schedule)  # 启用超级线程

    # 通过monitor 和 detector 测量的带宽和时延，设置graph的权重
    def create_weight_graph(self):
        # print(" update the bw and delay values ")
        # print("shortest--->=====>", self.network_monitor.port_free_bandwidth)
        self.network_monitor.create_bandwidth_graph()  # 调用monitor模块更新链路的带宽
        self.network_delay.create_delay_graph()  # 调用delay的方法更新链路的时延

    def save_weight(self):
        """保存权重的信息"""
        graph = self.network_structure.graph.copy()  # 拷贝一份网络拓扑的数据
        try:
            self.create_file()  # 创建文件
            WORK_DIR = Path.cwd().parent  # 获取本地父级目录
            txt_path = WORK_DIR / "ryu_code1/weight"
            pickle_name = '{}-{}'.format(self.count, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            txt_name = '{}-{}.txt'.format(self.count, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            self.save_pickle_graph(pickle_name, graph)
            with open(txt_path /  txt_name, 'w') as f:
                for i in graph.edges(data=True):
                    # dist = self.network_structure.ap_distance[(i[0], i[1])]
                    # i[2]['distance'] = dist
                    f.write(str(i) + "\n")
            self.count += 1
        except:
            print("the file is None")

    # def save_pickle_weight(self):
    #     """保存权重的信息"""
    #     graph = self.network_structure.graph.copy()  # 拷贝一份网络拓扑的数据
    #     try:
    #         WORK_DIR = route.cwd().parent  # 获取本地父级目录
    #         pickle_path = WORK_DIR / "ryu_code/weight_pickle"
    #
    #         with open(txt_path / '{}-{}.txt'.format(self.count, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())),
    #                   'w') as f:
    #             for i in graph.edges(data=True):
    #                 dist = self.network_structure.ap_distance[(i[0], i[1])]
    #                 i[2]['distance'] = dist
    #                 f.write(str(i) + "\n")
    #         self.count += 1
    #     except:
    #         print("the file is None")
    @staticmethod
    def save_pickle_graph(name, _graph):
        """
        保存成图信息的pkl文件
        /weight_pkl/xx-xx-xxx-xx-xx.pkl
        """
        WORK_DIR = Path.cwd().parent  # 获取本地父级目录
        pickle_path = WORK_DIR / "ryu_code1/weight_pickle"

        _path = pickle_path / Path(name + '.pkl')
        # _graph = self.network_structure.graph.copy()  # 拷贝一份网络拓扑的数据

        nx.write_gpickle(_graph, _path)  # 将文件写入图中

    def create_file(self):
        """创建一个新的文件夹"""
        WORK_DIR = Path.cwd().parent
        txt_path = WORK_DIR / "ryu_code1/weight"
        pickle_path = WORK_DIR / "ryu_code1/weight_pickle"
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
        if not os.path.exists(pickle_path):
            os.mkdir(pickle_path)

    def super_schedule(self):
        while True:
            hub.sleep(setting.SCHEDULE_PERIOD)
            # self.network_structure.scheduler()
            # self.network_structure.cal_shortest_path_thread()
            # self.network_monitor.scheduler()
            # self.network_delay.scheduler()
            # self.create_weight_graph()
            self.save_weight()  # 保存数据

    # 处理数据数据包发过来的数据
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # print("shortest------>PacketIn")
        msg = ev.msg
        datapath = msg.datapath
        # print("datapath.id------>", datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)
        # print("arp_pkt------>", arp_pkt)
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        # print("ipv4_pkt------>", ipv4_pkt)

        if isinstance(ipv4_pkt, ipv4.ipv4):
            print("shortest--->=====> IPv4 processing")
            if len(pkt.get_protocols(ethernet.ethernet)):
                eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
                # print("shortest-----------caleulate shortest path----------------")
                print("----------caleulate intelligent routing path-------")
                # 根据路径下发流表
                self.calculate_shortest_paths(msg, eth_type, ipv4_pkt.src, ipv4_pkt.dst)

    def calculate_shortest_paths(self, msg, eth_type, src_ip, dst_ip):
        """ 根据消息计算最短路径 """
        datapath = msg.datapath
        in_port = msg.match['in_port']

        # 1、找出位置
        # --------------------------------更改dpid的标识-----------------------------------#
        dpid = self.network_structure.apid_dict.get(datapath.id)
        src_dst_switches = self.get_switches(dpid, in_port, src_ip, dst_ip)
        # print("src_dst_switches ", src_dst_switches)
        if src_dst_switches:
            src_switch, dst_switch = src_dst_switches
            if src_switch:
                # 2、计算最短路径
                path = self.calculate_path(src_switch, dst_switch)
                # self.logger.info("shortest=====>[PATH] [%s <----> %s]: %s" % (src_ip, dst_ip, path))
                self.logger.info("Intelligent Routing:%s" % (path))
                # 3、下发流表
                self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data)
        else:
            self.logger.info("shortest---> src_dst_switches, 135", src_dst_switches)

    # 获得源交换机dpid和目标交换机的dpid
    def get_switches(self, dpid, in_port, src_ip, dst_ip):
        """ 根据src_ip 求得dpid """
        src_switch = dpid
        dst_switch = list()

        src_location = self.network_structure.get_host_ip_location(src_ip)  # (dpid, in_port)
        if in_port in self.network_structure.not_use_ports[dpid]:  # {dpid: {port_no, ...}}
            # print(f"shortest------>src_location == (dpid, in_port): {src_location} == {(dpid, in_port)}", )
            if (dpid, in_port) == src_location:
                src_switch = src_location[0]
            else:
                return None

        dst_location = self.network_structure.get_host_ip_location(dst_ip)
        if dst_location:
            dst_switch = dst_location[0]
        return src_switch, dst_switch

    def get_port(self, dst_ip):
        """根据目的ip获得出去的端口"""
        for key in self.network_structure.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if dst_ip == self.network_structure.access_table[key][0]:
                dst_port = key[1]
                return dst_port
        return None

    def get_port_pair(self, src_dpid, dst_dpid):
        """ 根据源dpid和目的dpid获得src.port_no, dst.port_no"""
        if (src_dpid, dst_dpid) in self.network_structure.link_port_table:
            return self.network_structure.link_port_table[(src_dpid, dst_dpid)]
        else:
            print("shortest--->dpid: %s -> dpid: %s is not in links", (src_dpid, dst_dpid))
            return None

    def calculate_path(self, src_dpid, dst_dpid):
        """计算最短路径"""
        self.network_structure.calculate_shortest_paths(src_dpid, dst_dpid, setting.WEIGHT)
        shortest_path = self.network_structure.shortest_path_table[(src_dpid, dst_dpid)]  # 获取最短路径
        return shortest_path

    def install_flow(self, path, eth_type, src_ip, dst_ip, in_port, buffer_id, data=None):
        """ 多种情况需要考虑 ， 根据条件判断走哪一个端口 """
        if path is None or len(path) == 0:
            print("shortest--->route Error")
            return
        else:
            first_dp = self.network_monitor.datapaths_table[path[0]]

            if len(path) > 2:
                # print("shortest--->len(path) > 2")
                for i in range(1, len(path) - 1):
                    port_pair = self.get_port_pair(path[i - 1], path[i])
                    port_pair_next = self.get_port_pair(path[i], path[i + 1])
                    # print("shortest--->len(path) > 2 port_pair, port_pair_next", port_pair, port_pair_next)
                    if port_pair and port_pair_next:
                        # TODO: 为什么port_pair[1]是src_port?  同一个交换机的不同口, 见图
                        src_port, dst_port = port_pair[1], port_pair_next[0]  # 同一个交换机的不同口, 见图
                        datapath = self.network_monitor.datapaths_table[path[i]]
                        # 下发正向流表
                        self.send_flow_mod(datapath, eth_type, src_ip, dst_ip, src_port, dst_port)
                        # 下发反向流表
                        self.send_flow_mod(datapath, eth_type, dst_ip, src_ip, dst_port, src_port)
                    else:
                        print(f"shortestERROR--->len(path) > 2 "
                              f"path_0, path_1, port_pair: {path[i - 1], path[i], port_pair}, "
                              f"path_1, path_2, next_port_pair: {path[i], path[i + 1], port_pair_next}")
                        return

            if len(path) > 1:
                # print("shortest--->len(path) == 2")
                port_pair = self.get_port_pair(path[-2], path[-1])

                if port_pair is None:
                    print("shortest--->port not found")
                    return

                src_port = port_pair[1]
                dst_port = self.get_port(dst_ip)

                if dst_port is None:
                    print("shortest--->Last port is not found")
                    return

                last_dp = self.network_monitor.datapaths_table[path[-1]]
                # TODO: 下发最后一个交换机的流表
                self.send_flow_mod(last_dp, eth_type, src_ip, dst_ip, src_port, dst_port)
                self.send_flow_mod(last_dp, eth_type, dst_ip, src_ip, dst_port, src_port)

                port_pair = self.get_port_pair(path[0], path[1])
                if port_pair is None:
                    print("shortest--->port not found in -2 switch")
                    return

                out_port = port_pair[0]
                # TODO: 发送倒数第二个交换机流表
                self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port)
                self.send_flow_mod(first_dp, eth_type, dst_ip, src_ip, out_port, in_port)
                self._build_packet_out(first_dp, buffer_id, in_port, out_port, data)

            else:
                out_port = self.get_port(dst_ip)
                if out_port is None:
                    print("shortest--->out_port is None in same dp")
                    return
                self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port)
                self.send_flow_mod(first_dp, eth_type, dst_ip, src_ip, out_port, in_port)
                self._build_packet_out(first_dp, buffer_id, in_port, out_port, data)

    def send_flow_mod(self, datapath, eth_type, src_ip, dst_ip, src_port, dst_port):
        """开始下发流表"""
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(dst_port)]

        match = parser.OFPMatch(in_port=src_port, eth_type=eth_type, ipv4_src=src_ip, ipv4_dst=dst_ip)

        self.add_flow(datapath, 1, match, actions)

    # 安装下发流表
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        ofp_parser = datapath.ofproto_parser

        inst = [ofp_parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority, idle_timeout=15,
                                        hard_timeout=60, match=match, instructions=inst)
        else:
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                        match=match, instructions=inst)
        datapath.send_msg(mod)

    # 构造输出的包
    def _build_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        actions = []  # 动作指令集
        if dst_port:
            actions.append(datapath.ofproto_parser.OFPActionOutput(dst_port))

        msg_data = None
        if buffer_id == datapath.ofproto.OFP_NO_BUFFER:
            if data is None:
                return None
            msg_data = data

        # send packet out msg  to datapath
        out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath, buffer_id=buffer_id,
                                                   in_port=src_port, actions=actions, data=msg_data)
        if out:
            datapath.send_msg(out)

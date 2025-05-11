# -*- coding: utf-8 -*-
"""
@File     : network_structure.py
@Date     : 2022-07-20 
@Author   : Terry_Li  -- 既然选择了远方，便只顾风雨兼程。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import copy
import time
from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.lib import hub
from ryu.lib.packet import packet, arp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link

import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import setting


class NetworkStructure(app_manager.RyuApp):
    """
    发现网络拓扑，保存网络结构
    """
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NetworkStructure, self).__init__(*args, **kwargs)
        self.name = 'discovery'
        self.topology_api_app = self
        self.graph = nx.Graph()
        self.pre_graph = nx.Graph()
        self.ap_distance = {}  # ap's distance
        self.link_info_xml = setting.LINKS_INFO  # xml file path of links info
        self.m_graph = self.parse_topo_links_info()  # 解析mininet构建的topo链路信息
        self.access_table = {}  # {(dpid, in_port): (src_ip, src_mac)}
        self.switch_all_ports_table = {}  # {dpid: {port_no, ...}}
        self.all_switches_dpid = self.switch_all_ports_table.keys()  # dict_key[dpid]
        self.switch_port_table = {}  # {dpid: {port, ...}
        self.link_port_table = {}  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}
        self.not_use_ports = {}  # {dpid: {port, ...}}  交换机之间没有用来连接的port
        self.shortest_path_table = {}  # {(src.dpid, dst.dpid): [path]}

        self.apid_dict = {}  # ap id dict 存放apid的字典
        self.apid_list = []  # ap id list

        self._discover_thread = hub.spawn(self.scheduler)
        self._shortest_path_thread = hub.spawn(self.cal_shortest_path_thread)

        self.first_flag = True
        self.cal_path_flag = False  # 计算路径的标志位

    def print_parameters(self):
        self.logger.info("discovery--->==================================== %s ====================================",
                         self.name)
        self.logger.info("discovery---> graph: %s", self.graph.edges)
        # self.logger.info("discovery---> access_table: %s", self.access_table)
        # self.logger.info("discovery---> switch_all_ports_table: %s", self.switch_all_ports_table)
        # self.logger.info("discovery---> switch_port_table: %s", self.switch_port_table)
        # self.logger.info("discovery---> link_port_table: %s", self.link_port_table)
        # self.logger.info("discovery---> not_use_ports: %s", self.not_use_ports)
        # self.logger.info("discovery---> shortest_path_table: %s", self.shortest_path_table)
        self.logger.info("discovery--->=============================================================================")

    def scheduler(self):
        i = 0
        while True:
            if i == 3:
                self.get_topology(None)
                i = 0
            hub.sleep(setting.DISCOVERY_PERIOD)
            if setting.PRINT_SHOW:
                self.print_parameters()
            i += 1

    def cal_shortest_path_thread(self):
        """ 计算所有路径的最短距离 """
        # self.cal_path_flag = False
        # while True:
        #     if self.cal_path_flag:
        self.calculate_all_nodes_shortest_paths(weight=setting.WEIGHT)
        # hub.sleep(setting.DISCOVERY_PERIOD)

    # Flow mod and Table miss
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # ------------------------给dpid做映射处理-----------------------------------#
        self.apid_list.append(datapath.id)  # 将id存进列表中，目的是获取id的个数
        self.apid_list.sort()  # 修改，对列表元素进行排序处理
        for t in range(1, len(self.apid_list) + 1):
            self.apid_dict[self.apid_list[t - 1]] = t + 3 # 将id存进字典中

        self.logger.info("discovery---> AccessPoint: %s connected", self.apid_dict.get(datapath.id))

        # install table miss flow entry
        match = parser.OFPMatch()  # match all
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]

        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        inst = [datapath.ofproto_parser.OFPInstructionActions(datapath.ofproto.OFPIT_APPLY_ACTIONS,
                                                              actions)]
        mod = datapath.ofproto_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                                 match=match, instructions=inst)
        datapath.send_msg(mod)

    # Packet In
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # print("discovery---> discovery PacketIn")
        msg = ev.msg
        datapath = msg.datapath
        # ofproto = datapath.ofproto
        # parser = datapath.ofproto_parser

        # 输入端口号
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)

        # if arp_pkt:
        if isinstance(arp_pkt, arp.arp):
            # self.logger.info("discovery---> arp packet")
            arp_src_ip = arp_pkt.src_ip
            src_mac = arp_pkt.src_mac
            self.storage_access_info(self.apid_dict.get(datapath.id), in_port, arp_src_ip, src_mac)
            # print("11111---->",self.access_table)

    # 将packet-in解析的arp的网络通路信息存储
    def storage_access_info(self, dpid, in_port, src_ip, src_mac):
        if in_port in self.not_use_ports[dpid]:
            # print("discovery--->", dpid, in_port, src_ip, src_mac)
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (src_ip, src_mac):
                    return
                else:
                    self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                    return
            else:
                self.access_table.setdefault((dpid, in_port), None)
                self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                return

    # 利用topology库获取拓扑信息
    events = [event.EventSwitchEnter, event.EventSwitchLeave,
              event.EventPortAdd, event.EventPortDelete, event.EventPortModify,
              event.EventLinkAdd, event.EventLinkDelete]

    @set_ev_cls(events)
    def get_topology(self, ev):
        # self.logger.info("discovery--->-----> EventSwitch/Port/Link")
        # 事件发生时，获得swicth列表
        switch_list = get_switch(self.topology_api_app, None)
        # 将swicth添加到self.switch_all_ports_table
        # 更改了dpid的方式
        for switch in switch_list:
            dpid = switch.dp.id
            self.switch_all_ports_table.setdefault(self.apid_dict.get(dpid), set())
            self.switch_port_table.setdefault(self.apid_dict.get(dpid), set())
            self.not_use_ports.setdefault(self.apid_dict.get(dpid), set())

            for p in switch.ports:
                self.switch_all_ports_table[self.apid_dict.get(dpid)].add(p.port_no)

        self.all_switches_dpid = self.switch_all_ports_table.keys()

        # 获得link
        link_list = get_link(self.topology_api_app, None)
        self.link_port_table = {}
        # 将link添加到self.link_table
        # -------------------更改---------------
        # --------------把所有的dpid改成映射的id--------------------#
        for link in link_list:
            src = link.src
            dst = link.dst
            self.link_port_table[(self.apid_dict.get(src.dpid), self.apid_dict.get(dst.dpid))] = (
                src.port_no, dst.port_no)

            if self.apid_dict.get(src.dpid) in self.all_switches_dpid:
                self.switch_port_table[self.apid_dict.get(src.dpid)].add(src.port_no)
            if self.apid_dict.get(dst.dpid) in self.all_switches_dpid:
                self.switch_port_table[self.apid_dict.get(dst.dpid)].add(dst.port_no)

        # 统计没使用的端口
        for sw_dpid in self.switch_all_ports_table.keys():
            all_ports = self.switch_all_ports_table[sw_dpid]
            linked_port = self.switch_port_table[sw_dpid]
            # print("discovery---> all_ports, linked_port", all_ports, linked_port)
            self.not_use_ports[sw_dpid] = all_ports - linked_port

        # 建立拓扑 bw和delay未定
        self.build_topology_between_switches()
        # self.cal_path_flag = True

    def build_topology_between_switches(self, free_bw=0, delay=0, loss=0, used_bw=0, pkt_err=0, pkt_drop=0, distance=0):
        """ 根据 src_dpid 和 dst_dpid 建立拓扑，bw 和 delay 信息还未定"""
        _graph = nx.Graph()

        # self.graph.clear()
        for (src_dpid, dst_dpid) in self.link_port_table.keys():
            # 建立switch之间的连接，端口可以通过查link_port_table获得
            _graph.add_edge(src_dpid, dst_dpid, free_bw=free_bw, delay=delay, loss=loss,
                            used_bw=used_bw, pkt_err=pkt_err, pkt_drop=pkt_drop, distance=distance)
        if _graph.edges == self.graph.edges:
            return
        else:
            self.graph = _graph

    def calculate_weight(self, node1, node2, weight_dict):
        """ 计算路径时，weight可以调用函数，该函数根据因子计算 bw * factor - delay * (1 - factor) 后的weight"""
        # weight可以调用的函数
        assert 'bw' in weight_dict and 'delay' in weight_dict, "edge weight should have bw and delay"
        try:
            weight = weight_dict['bw'] * setting.FACTOR - weight_dict['delay'] * (1 - setting.FACTOR)
            return weight
        except TypeError:
            print("discovery ERROR---->weight_dict['bw']:", weight_dict['bw'])
            print("discovery ERROR---->weight_dict['delay']:", weight_dict['delay'])
            return None

    def calculate_shortest_paths(self, src_dpid, dst_dpid, weight=None):
        """ 计算src到dst的最短路径，存在self.shortest_path_table中"""
        # TODO: 应该深拷贝，防止没算出来时改变graph
        # try:
        graph = self.graph.copy()
        self.shortest_path_table[(src_dpid, dst_dpid)] = nx.shortest_path(self.graph, src_dpid, dst_dpid,
                                                                          weight=weight, method=setting.METHOD)
        # except TypeError:
        #     self.shortest_path_table[(src_dpid, dst_dpid)] = None
        #     self.logger.info("discovery--->TypeError not found path between %d and %d", src_dpid, dst_dpid)

    def calculate_all_nodes_shortest_paths(self, weight=None):
        """ 根据已构建的图，计算所有nodes间的最短路径，weight为权值，可以为calculate_weight()该函数"""
        self.shortest_path_table = {}  # 先清空，再计算
        for src in self.graph.nodes():
            for dst in self.graph.nodes():
                if src != dst:
                    self.calculate_shortest_paths(src, dst, weight=weight)
                else:
                    continue

    def get_host_ip_location(self, host_ip):
        """
            通过host_ip查询 self.access_table: {(dpid, in_port): (src_ip, src_mac)}
            获得(dpid, in_port)
        """

        for key in self.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if self.access_table[key][0] == host_ip:
                # print("discovery--->zzzz---> key", key)
                return key
        # FIXME: 刚开始这里写成else了，导致一直循环
        # print("discovery--->%s location is not found" % host_ip)
        return None

    def parse_topo_links_info(self):
        """解析拓扑信息"""
        m_graph = nx.Graph()
        parser = ET.parse(self.link_info_xml)
        root = parser.getroot()

        links_info_element = root.find("links_info")

        def _str_tuple2int_list(s: str):
            s = s.strip()
            assert s.startswith('(') and s.endswith(")"), '应该为str的元组，如 "(1, 2)"'
            s_ = s[1: -1].split(', ')
            return [int(i) for i in s_]

        node1, node2, port1, port2, bw, delay, loss, distance = None, None, None, None, None, None, None, None
        for e in root.iter():
            if e.tag == 'links':
                node1, node2 = _str_tuple2int_list(e.text)
            elif e.tag == 'ports':
                port1, port2 = _str_tuple2int_list(e.text)
            elif e.tag == 'bw':
                bw = float(e.text)
            elif e.tag == 'delay':
                delay = float(e.text[:-2])
            elif e.tag == 'loss':
                loss = float(e.text)
            elif e.tag == 'distance':
                distance = float(e.text)
            else:
                print(e.tag)
                continue
            # self.ap_distance[(node1, node2)] = distance
            # self.ap_distance[(node2, node1)] = distance
            m_graph.add_edge(node1, node2, port1=port1, port2=port2, free_bw=bw, delay=delay, loss=loss,
                             used_bw=0, pkt_err=0, pkt_drop=0, distance=distance)

        for edge in m_graph.edges(data=True):
            print(edge)
        return m_graph

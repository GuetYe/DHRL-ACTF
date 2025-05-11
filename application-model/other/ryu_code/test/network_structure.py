# -*- coding: utf-8 -*-
"""
@File     : network_structure.py
@Date     : 2022-07-20 
@Author   : Terry_Li  -- 既然选择了远方，便只顾风雨兼程。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import arp, ethernet, packet
from ryu.topology.api import get_switch, get_link

import setting
import copy
import networkx as nx
import matplotlib.pyplot as plt
from ryu.topology import event
from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3


class NetworkStructure(app_manager.RyuApp):
    """Network structure manager  发现网络拓扑，保存网络结构"""

    # 指定OpenFlow13 版本
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NetworkStructure, self).__init__(*args, **kwargs)
        self.name = 'discovery'  # 给该类取个名字方便后面进行调用
        self.topology_api_app = self
        self.graph = nx.DiGraph()  # 定义一个空的有向图用于存储链路信息
        self.pre_graph = nx.DiGraph()
        self.not_use_ports = {}  # {dpid:{port,...}} 交换机之间没有用来连接的port
        self.access_table = {}  # {(dpid, in_port): (src_ip, src_mac)}
        self.switch_all_ports_table = {}  # {dpid: {port_no, ...}}
        self.all_switches_dpid = self.switch_all_ports_table.keys()  # dict_key[dpid]
        self.switch_port_table = {}  # {dpid: {port, ...}
        self.link_port_table = {}  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}
        self.shortest_path_table = {}  # {(src.dpid, dst.dpid): [path]}

        # self._discover_thread = hub.spawn(self._discover_network_structures)
        # self._show_graph = hub.spawn(self.show_graph_plt())
        # self.scheduler_thread = hub.spawn(self.scheduler)

    def print_parameters(self):
        self.logger.info("discovery --------->=========================%s=====================", self.name)
        self.logger.info("discovery --------->graph: %s", self.graph.edges)
        self.logger.info("discovery ------> ==================================")

    def _discover_network_structures(self):
        """发现网络结构的函数"""
        first_flag = setting.FIRST_FLAG  # True
        while True:
            hub.sleep(setting.DISCOVERY_PERIOD)
            self.get_topology(None)
            if self.pre_graph.edges != self.graph.edges or first_flag:
                self.print_parameters()
                first_flag = False

    def scheduler(self):
        """进行线程调度"""
        self.get_topology(None)
        if setting.PRINT_SHOW or setting.FIRST_FLAG:
            self.first_flag = False
            self.print_parameters()

    # Flow mod and  Table miss
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        datapath中有配置消息到达
        """
        datapath = ev.msg.datapath  # 输出 <ryu.controller.controller.Datapath object at 0x7f1e53b22400>
        ofproto = datapath.ofproto  # <module 'ryu.ofproto.ofproto_v1_3>
        parser = datapath.ofproto_parser  # <module 'ryu.ofproto.ofproto_v1_3_parser>

        self.logger.info("discovery ---> switch :  %s connected" % datapath.id)
        # discovery ---> switch :  1 connected
        # discovery ---> switch :  2 connected
        # discovery ---> switch :  3 connected

        #  install  table miss  flow entry
        match = parser.OFPMatch()  # OFPMatch(oxm_fields={})
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]  # [OFPActionOutput(len=16,max_len=65535,port=4294967293,type=0)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        """Add a flow"""
        inst = [datapath.ofproto_parser.OFPInstructionActions(datapath.ofproto.OFPIT_APPLY_ACTIONS,
                                                              actions)]
        # print  [OFPInstructionActions(actions=[OFPActionOutput(len=16,max_len=65535,port=4294967293,type=0)],type=4)]
        mod = datapath.ofproto_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                                 match=match, instructions=inst)
        datapath.send_msg(mod)

    # Packet In
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # print("discovery------> discovery Packet In")
        # 处理消息的函数，只有在mininet中进行pingall 才能运行这个函数
        msg = ev.msg
        # version=0x4,msg_type=0xa,msg_len=0x70,xid=0x0,
        # OFPPacketIn(buffer_id=4294967295,cookie=0,
        # data=b'33\x00\x00\x00\x02\x00\x00\x00\x00\x00\x02\x86\xdd`\x00\x00\x00\x00\x10:\xff\xfe\x80\x00\x00\x00\x00\x00\x00\x02\x00\x00\xff\xfe\x00\x00\x02\xff\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x85\x00{*\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x02',
        # match=OFPMatch(oxm_fields={'in_port': 2}),reason=0,table_id=0,total_len=70)
        datapath = msg.datapath
        ofproto = datapath.ofproto
        paser = datapath.ofproto_parser

        # 输入端口号
        in_port = msg.match["in_port"]  # 数字
        # 解析数据包的网络协议
        # ethernet(dst='ff:ff:ff:ff:ff:ff',ethertype=2054,src='00:00:00:00:00:01'),
        # arp(dst_ip='10.0.0.3',dst_mac='00:00:00:00:00:00',
        # hlen=6,hwtype=1,opcode=1,plen=4,proto=2048,src_ip='10.0.0.1',
        # src_mac='00:00:00:00:00:01')
        pkt = packet.Packet(msg.data)
        # arp_pkt = pkt.get_protocols(ethernet.ethernet)
        #  [ethernet(dst='ff:ff:ff:ff:ff:ff',ethertype=2054,src='00:00:00:00:00:01')]
        arp_pkt = pkt.get_protocol(arp.arp)
        # [arp(dst_ip='10.0.0.1',dst_mac='00:00:00:00:00:00',hlen=6,hwtype=1,opcode=1,
        # plen=4,proto=2048,src_ip='10.0.0.2',src_mac='00:00:00:00:00:02')]
        if arp_pkt:
            # self.logger.info("discovery----> arp packet")
            arp_src_ip = arp_pkt.src_ip  # 10.0.0.1 获取源头ip地址
            src_mac = arp_pkt.src_mac  # 00:00:00:00:00:01  获取源mac地址
            self.storage_access_info(datapath.id, in_port, arp_src_ip, src_mac)

        # 将packet- in 解析的arp的网络通路信息进行存储

    def storage_access_info(self, dpid, in_port, src_ip, src_mac):
        # if in_port in self.not_use_ports:
        #     # print("discovery------>",dpid,in_port,src_ip,src_mac)
        #     # print("discover------>before-->",self.access_table)

        #     if (dpid,in_port) not in self.access_table.keys() or self.access_table[(dpid, in_port)] != (src_ip,src_mac):
        #         self.access_table[(dpid, in_port)] = (src_ip, src_mac) # 采用dipd做key
        #         # print("discover----->discover after-->", self.access_table)
        #     else:
        #         print("discovery------>network access already exits", end="", flush=False)
        #         print(self.access_table)

        # else:
        #     self.logger.info("discovery--->in_port can't use")
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

    # 利用topology 库来获取拓扑信息
    events = [event.EventSwitchEnter, event.EventSwitchLeave,
              event.EventSwitchReconnected,
              event.EventPortAdd, event.EventPortDelete,
              event.EventPortModify,
              event.EventLinkAdd, event.EventLinkDelete]

    # 获取链路信息
    @set_ev_cls(events)
    def get_topology(self, ev):
        # self.logger.info("discovery--->-----> EventSwitch/Port/Link")
        # 事件发生时，获得swicth列表
        switch_list = get_switch(self.topology_api_app, None)
        # 将swicth添加到self.switch_all_ports_table
        for switch in switch_list:
            dpid = switch.dp.id
            self.switch_all_ports_table.setdefault(dpid, set())
            self.switch_port_table.setdefault(dpid, set())
            self.not_use_ports.setdefault(dpid, set())

            for p in switch.ports:
                self.switch_all_ports_table[dpid].add(p.port_no)

        # 获得link
        link_list = get_link(self.topology_api_app, None)
        self.link_port_table = {}
        # 将link添加到self.link_table
        for link in link_list:
            src = link.src  # 实际是个port实例，我找了半天
            dst = link.dst
            self.link_port_table[(src.dpid, dst.dpid)] = (src.port_no, dst.port_no)

            if src.dpid in self.all_switches_dpid:
                self.switch_port_table[src.dpid].add(src.port_no)
            if dst.dpid in self.all_switches_dpid:
                self.switch_port_table[dst.dpid].add(dst.port_no)

        # 统计没使用的端口
        for sw_dpid in self.switch_all_ports_table.keys():
            all_ports = self.switch_all_ports_table[sw_dpid]
            linked_port = self.switch_port_table[sw_dpid]
            # print("discovery---> all_ports, linked_port", all_ports, linked_port)
            self.not_use_ports[sw_dpid] = all_ports - linked_port

        # 建立拓扑 bw 和 delay 未定
        self.build_topology_between_switches()

    def build_topology_between_switches(self, bw=None, delay=None):
        """ 根据 src_dpid 和 dst_dpid 建立拓扑，bw 和 delay 信息还未定"""
        # networkx使用已有Link的src_dpid 和 dst_dpid 信息建立拓扑
        self.pre_graph = copy.deepcopy(self.graph)
        self.graph.clear()
        for (src_dpid, dst_dpid) in self.link_port_table.keys():
            # 建立switch之间的连接，端口可以通过查link_port_table获得
            self.graph.add_edge(src_dpid, dst_dpid, bw=bw, delay=delay)

    def calculate_weight(self, node1, node2, weight_dict):
        """计算路径时，weight可以调用函数，该函数根据因子计算bw*factor-delay*(1-factor) 后的weight"""
        #  weight 可以调用的函数
        assert 'bw' in weight_dict and 'delay' in weight_dict, "edge weight should have bw and delay"
        weight = weight_dict['bw'] * setting.FACTOR - weight_dict['delay'] * (1 - setting.FACTOR)

    def calculate_shortest_paths(self, src_dpid, dst_dpid, weight=None):
        """计算src到dst的最短路径，存在self.shortest_path_table中"""
        # TODO ：应该深度拷贝，防止没算出来时改变graph
        graph = self.graph.copy()
        self.shortest_path_table[(src_dpid, dst_dpid)] = nx.shortest_path(self.graph, src_dpid, dst_dpid, weight=weight,
                                                                          method=setting.METHOD)

    def calculate_all_shortest_paths(self, weight=None):
        """根据已构建的图，计算所有nodes间的最短路径，weight为权值，可以采用calculate_weight()进行计算"""
        self.shortest_path_table = {}  # 先清空再进行计算
        for src in self.graph.nodes():
            for dst in self.graph.nodes():
                if src != dst:
                    self.calculate_shortest_paths(src, dst, weight=weight)
                else:
                    continue

    def get_host_ip_location(self, host_ip):
        """
           通过host_ip查询self.acess_table: {(dpid, in_port): (src_ip, src_mac)}
           获得（dpid, in_port)
        """
        for key in self.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if self.access_table[key][0] == host_ip:
                # print("discovery----->ZZZZ---->key",key)
                return key

        print("discovery------>%s location is not  found " % host_ip)

        return None

    def show_graph_plt(self):
        FIRST = True
        if self.pre_graph != self.graph or FIRST:
            FIRST = False
            nx.draw_networkx(self.graph, with_labels=True)
            plt.show()

# -*- coding: utf-8 -*-
"""
@File     : my_shortest_forwarding.py
@Date     : 2022-07-20 16:06
@Author   : Terry_Li  - 长风破浪会有时，直挂云帆济沧海。
IDE       : PyCharm
@Mail     : terry.li.dev@foxmail.com
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, set_ev_cls, MAIN_DISPATCHER
from ryu.lib.packet import packet, arp
from ryu.lib.packet import ethernet, ipv6
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import networkx as nx
from ryu.lib import mac
from ryu.base.app_manager import  lookup_service_brick
from ryu.lib import hub
import setting


class ExampleShortestForwarding(app_manager.RyuApp):
    """string for description"""
    # openflow version is 1.3
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ExampleShortestForwarding, self).__init__(*args, **kwargs)
        self.network = nx.DiGraph()  # network graph
        self.topology_api_app = self
        self.delay = lookup_service_brick("delay")
        self.name = 'shortest_path_forwarding'
        self.paths = {}
         # 定义MAC地址列表，这里的mac_to_port表就是对应的交换机二层通信查询表
        self.mac_to_port = {}
        self.arp_table = {}
        self.sw = {}
        self.shortest_thread = hub.spawn(self.super_schedule) # 启用超级协程

    def super_schedule(self):
        """超级协程"""
        while True:
            hub.sleep(setting.SCHEDULE_PERIOD)
            self.delay._detector()
            # self.discovery.scheduler()
            # self.monitor.scheduler()
            # self.detector.scheduler()
            # self.create_weight_graph()

    # handle switch features in packets
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        ofp_parser = datapath.ofproto_parser

        # install  a table-miss flow entry for each datapath
        match = ofp_parser.OFPMatch()
        actions = [ofp_parser. OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                               ofproto.OFPCML_NO_BUFFER)]
        # install flow table
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("switch: %s connected", datapath.id)

     # 安装下发流表
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        ofp_parser = datapath.ofproto_parser

        inst = [ofp_parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority, idle_timeout=5,
                                        hard_timeout=15, match=match, instructions=inst)
        else:
            mod = ofp_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                        match=match, instructions=inst)
        datapath.send_msg(mod)

     # get topology information and store it into networkx object
    @set_ev_cls(event.EventSwitchEnter, [CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def get_topology(self, ev):
        # get nodes
        switch_list = get_switch(self.topology_api_app, None)
        switches = [switch.dp.id for switch in switch_list]  # del self
        self.network.add_nodes_from(switches)  # add nodes to switch list

        # get links
        link_list = get_link(self.topology_api_app, None)
        links = [(link.src.dpid, link.dst.dpid,
                 {'port': link.src.port_no}) for link in link_list]
        self.network.add_edges_from(links)  # add links to switch

        # get  reverse links
        links = [(link.dst.dpid, link.src.dpid,
                 {'port': link.dst.port_no}) for link in link_list]
        self.network.add_edges_from(links)  # add links to switch

     # get out_port by  using  networkx's Dijkstra  algorithm
    def get_out_port(self, datapath, src, dst, in_port):
        dpid = datapath.id
        # add links between host and acess switch
        if src not in self.network:
            self.network.add_node(src)
            self.network.add_edge(dpid, src, port=in_port)
            self.network.add_edge(src, dpid)
            self.paths.setdefault(src, {})

        # search dst 's shortest path
        if dst in self.network:
            if dst not in self.paths[src]:
                path = nx.shortest_path(self.network, src, dst)
                self.paths[src][dst] = path

            path = self.paths[src][dst]  # [1,2,3]
            next_hop = path[path.index(dpid) + 1]
            out_port = self.network[dpid][next_hop]['port']

            print("The shortest path is:")
            print("path", path)
        else:
            out_port = datapath.ofproto.OFPP_FLOOD  # 泛洪处理

        return out_port

        # handle packet in  msg

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
         数据包进入：消息处理
         :param ev: 事件
         :return:
         """
        # 1. 初始设置
        msg = ev.msg  # 监听到事件的消息
        datapath = msg.datapath  # 数据平面的通道
        ofproto = datapath.ofproto  # OpenFlow的版本
        ofp_parser = datapath.ofproto_parser  # OpenFlow解析的库类

            # 2. 获取数据平面通道的ID，并存储信息
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

            # 3. 解析和分析收到的数据包
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        dst = eth_pkt.dst
        src = eth_pkt.src
        in_port = msg.match['in_port']

        # self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

            # 4. 学习mac到port的信息:记录进字典内
        self.mac_to_port[dpid][src] = in_port

            # 5. 查询转发的目的地址是否在学习到的表中，若不在：泛洪
        if dst in self.mac_to_port[dpid]:
                # out_port = self.mac_to_port[dpid][dst]
                # get out_port path info
                out_port = self.get_out_port(datapath, eth_pkt.src, eth_pkt.dst, in_port)
        else:
                out_port = ofproto.OFPP_FLOOD

        actions = [ofp_parser.OFPActionOutput(out_port)]

            # install flow entries
        if out_port != ofproto.OFPP_FLOOD:
                match = ofp_parser.OFPMatch(in_port=in_port, eth_dst=eth_pkt.dst)
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
                data = msg.data

            # send packet out msg  to datapath
        out = ofp_parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                          in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

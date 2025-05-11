# -*- coding: utf-8 -*-
"""
@File     : network_monitor.py
@Date     : 2022-07-21 
@Author   : Terry_Li  -- 逍遥半生酒中意，一剑碎影向征程。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
from operator import attrgetter

import setting
from ryu.base.app_manager import lookup_service_brick
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3


class NetworkMonitor(app_manager.RyuApp):
    """监控网络流量状态"""
    # TODO：openflow1.3版本
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NetworkMonitor, self).__init__(*args, **kwargs)
        self.name = 'monitor'
        self.datapaths_table = {}  # {dpid: datapath}
        self.dpid_port_fueatures_table = {}  # {dpid:{port_no: (config, state, curr_speed, max_speed)}}

        self.port_stats_table = {}  # {(dpid, port_no): [(stat.tx_bytes, stat.rx_bytes, stat.rx_errors,stat.duration_sec, stat.duration_nsec), .....]}
        self.flow_stats_table = {}  # {dpid:{(in_port, ipv4_dsts, out_port): (packet_count, byte_count, duration_sec, duration_nsec)}}
        self.port_speed_table = {}  # {(dpid, port_no): [speed, .....]}
        self.flow_speed_table = {}  # {dpid: {(in_port, ipv4_dsts, out_port): speed}}
        self.port_flow_dpid_stats = {'port': {}, 'flow': {}}
        self.port_curr_speed = {}  # {dpid: {port_no: curr_bw}}

        self.port_loss = {} # loss value

        self.port_pkt_drop = {} # 弃包个数
        self.port_pkt_err = {} # 错包率

        self.network_structure = lookup_service_brick("discovery")  # 创建一个NetworkStructure的实例

        self.monitor_thread = hub.spawn(self.scheduler)
        # self.monitor_thread = hub.spawn(self.print_parameters)

    def print_parameters(self):
        self.logger.info("monitor---->===================================%s=================================",
                         self.name)
        print("monitor\n----->self.datapaths_table", self.datapaths_table)
        print("monitor\n----->self.dpid_port_fueatures_table", self.dpid_port_fueatures_table)
        print("monitor\n----->self.port_stats_table", self.port_stats_table)
        print("monitor\n----->self.flow_stats_table", self.flow_stats_table)
        print("monitor\n----->self.port_speed_table", self.port_speed_table)
        print("monitor\n----->self.flow_speed_table", self.flow_speed_table)
        print("monitor\n----->self.port_curr_speed ", self.port_curr_speed)
        self.logger.info("===================monitor=======================")

    def scheduler(self): 
        hub.sleep(10)
        while True:
            hub.sleep(setting.MONITOR_PERIOD)
            # print("monitor\n----->self.datapaths_table", self.datapaths_table)
            self._request_stats()
            self.create_bandwidth_graph()
            self.update_graph_loss()
        # if True:
        #     self.print_parameters() # 打印测试的参数

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """存放所有的datapath实例"""
        datapath = ev.datapath  # EventOFPStateChange类可以直接获得datapath
        # print("datapath---->", datapath)
        # 当运行mininet拓扑信息时将会进行交换机的注册
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths_table:
                self.logger.info("monitor---->register datapath: %016x", datapath.id)  # 打印交换机的掩码地址
                #-----------------------------更改dpid的标识---------------------------------------#
                dpid =  self.network_structure.apid_dict.get(datapath.id)
                self.datapaths_table[dpid] = datapath
               
                # 一些初始化
                self.dpid_port_fueatures_table.setdefault(dpid, {})
                self.flow_stats_table.setdefault(dpid, {})
                # print("1111111", self.dpid_port_fueatures_table)
        # 当退出mininet后将提示交换机未注册，同时删除datapaths_table 字典里面的数据
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths_table:
                self.logger.info("monitor---> unreigster datapath: %016x", datapath.id)
                del self.datapaths_table[datapath.id]

    # 主动发送request, 请求状态信息
    def _request_stats(self):
        # print("monitor--->send request --->  ---> send reuest ---> ")
        for datapath in self.datapaths_table.values():
            # self.logger.info("monitor-----> send stats request: %016x", datapath.id)
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser

            # 1、端口描述请求
            req = parser.OFPPortDescStatsRequest(datapath, 0)
            datapath.send_msg(req)

            # 2、端口统计请求
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)  # 所有端口
            datapath.send_msg(req)

            # 3、单个流统计请求
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)

    # 处理上面请求的回复OFPPortDescStatsReoly
    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        """ 存储端口描述信息， 见OFPPort类，配置，状态、当前速度 """
        # print("monitor---> EventOFPPortDescStatsReply")
        msg = ev.msg
        datapath = msg.datapath
        #-------------------------------- 更改dpid的标识------------------------#
        dpid = self.network_structure.apid_dict.get(datapath.id)  # 交换机的id标识
        # print("dpid-------->",dpid)
        ofproto = msg.datapath.ofproto

        config_dict = {ofproto.OFPPC_PORT_DOWN: 'Port Down',
                       ofproto.OFPPC_NO_RECV: 'No Recv',
                       ofproto.OFPPC_NO_FWD: 'No Forward',
                       ofproto.OFPPC_NO_PACKET_IN: 'No Pakcet-In'}
        # config_dict--------> {1: 'Port Down', 4: 'No Recv', 32: 'No Forward', 64: 'No Pakcet-In'}
        # print("config_dict-------->",config_dict)

        state_dict = {ofproto.OFPPS_LINK_DOWN: "Link Down",
                      ofproto.OFPPS_BLOCKED: "Blocked",
                      ofproto.OFPPS_LIVE: "Live"}
        # state_dict--------> {1: 'Link Down', 2: 'Blocked', 4: 'Live'}
        # print("state_dict-------->", state_dict)

        for ofport in ev.msg.body:
            if ofport.port_no != ofproto_v1_3.OFPP_LOCAL:  # 0xfffffffe 4294967294

                if ofport.config in config_dict:
                    config = config_dict[ofport.config]
                else:
                    config = 'Up'

                if ofport.state in state_dict:
                    state = state_dict[ofport.state]
                else:
                    state = 'Up'

                # 存储配置、状态、curr_speed,max_speed = 0
                port_features = (config, state, ofport.curr_speed, ofport.max_speed)
                self.dpid_port_fueatures_table[dpid][ofport.port_no] = port_features

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_table_reply_handler(self, ev):
        """ 存储端口统计信息， 见OFPPortStats, 发送bytes、接收bytes、生效时间duration_sec等"""
        # print("monitor----> EventOFPPortStatsReply")
        body = ev.msg.body
        # ----------------------------更改dpid-------------------------#
        dpid = self.network_structure.apid_dict.get(ev.msg.datapath.id)
        self.port_flow_dpid_stats['port'][dpid] = body
        # print("self.port_flow_dpid_stats",self.port_flow_dpid_stats)

        # 打印流量信息
        if setting.PRINT_SHOW:
            print("port --------------->流量监控\n")
            self.logger.info('datapath         port     '
                             'rx-pkts  rx-bytes rx-error '
                             'tx-pkts  tx-bytes tx-error')
            self.logger.info('---------------- -------- '
                             '-------- -------- -------- '
                             '-------- -------- --------')

        for stat in sorted(body, key=attrgetter("port_no")):
            port_no = stat.port_no
            if port_no != ofproto_v1_3.OFPP_LOCAL:
                if setting.PRINT_SHOW:
                    self.logger.info('%016x %8x %8d %8d %8d %8d %8d %8d',
                                     dpid, stat.port_no,
                                     stat.rx_packets, stat.rx_bytes, stat.rx_errors,
                                     stat.tx_packets, stat.tx_bytes, stat.tx_errors)
                key = (dpid, port_no)
                value = (stat.tx_bytes, stat.rx_bytes, stat.rx_errors,
                         stat.duration_sec, stat.duration_nsec, stat.tx_packets, stat.rx_packets)
                # print("key-------->", key)
                # print("value-------->", value)
                self._save_stats(self.port_stats_table, key, value, 5)  # 保存信息， 最多保存前5次

                pre_bytes = 0
                delta_time = setting.MONITOR_PERIOD  # 每10s检测一次网络流量状态
                stats = self.port_stats_table[key]  # 获得已经存了的统计信息
                # print("stats-------->", stats)

                if len(stats) > 1:  # 有两次以上的信息
                    pre_bytes = stats[-2][0] + stats[-2][1]
                    # print("pre_bytes------>", pre_bytes)
                    delta_time = self._calculate_delta_time(stats[-1][3], stats[-1][4],
                                                            stats[-2][3], stats[-2][4])  # 倒数第一个统计信息，倒数第二个统计信息
                    # print("delta_time------>",delta_time)
                speed = self._calculate_speed(stats[-1][0] + stats[-1][1],
                                              pre_bytes, delta_time)
                self._save_stats(self.port_speed_table, key, speed, 5)
                self._calculate_port_speed(dpid, port_no, speed)

        self.calculate_loss_of_link()

        # print("\n")

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """ 存储flow的状态，计算pingall之后流表的流速等状态信息"""
        msg = ev.msg
        body = msg.body
        datapath = msg.datapath
        # -------------------------更改dpid----------------------#
        dpid = self.network_structure.apid_dict.get(datapath.id)

        self.port_flow_dpid_stats['flow'][dpid] = body
        # print("monitor---> body", body)

        # 打印流量信息
        if setting.PRINT_SHOW:
            print("flow --------------->流量监控\n")
            self.logger.info('datapath         '
                             'in-port  eth-dst           '
                             'out-port packets  bytes')
            self.logger.info('---------------- '
                             '-------- ----------------- '
                             '-------- -------- --------')

        for stat in sorted([flowstats for flowstats in body if flowstats.priority == 1],
                           key=lambda flowstats: (flowstats.match.get('in_port'), flowstats.match.get('ipv4_dst'))):
            if setting.PRINT_SHOW:
                self.logger.info('%016x %8x %17s %8x %8d %8d',
                                 dpid,
                                 stat.match['in_port'], stat.match['ipv4_dst'],
                                 stat.instructions[0].actions[0].port,
                                 stat.packet_count, stat.byte_count)
            # print("monitor---> stat.match", stat.match)
            # print("monitor---> stat", stat)
            key = (stat.match['in_port'], stat.match['ipv4_dst'],
                   stat.instructions[0].actions[0].port)
            value = (stat.packet_count, stat.byte_count, stat.duration_sec, stat.duration_nsec)
            self._save_stats(self.flow_stats_table[dpid], key, value, 5)

            pre_bytes = 0
            delta_time = setting.MONITOR_PERIOD
            value = self.flow_stats_table[dpid][key]
            if len(value) > 1:
                pre_bytes = value[-2][1]
                delta_time = self._calculate_delta_time(value[-1][2], value[-1][3],
                                                        value[-2][2], value[-2][3])
            speed = self._calculate_speed(self.flow_stats_table[dpid][key][-1][1], pre_bytes, delta_time)
            self.flow_speed_table.setdefault(dpid, {})
            self._save_stats(self.flow_speed_table[dpid], key, speed, 5)

            
        self.calculate_loss_of_link()

        # print("\n")

    # 存多次数据，比如一个端口存上一次的统计信息和这一次的统计信息
    @staticmethod
    def _save_stats(_dict, key, value, keep):
        if key not in _dict:
            _dict[key] = []
        _dict[key].append(value)

        if len(_dict[key]) > keep:
            _dict[key].pop(0)  # 弹出最早的数据


    def _calculate_delta_time(self, now_sec, now_nsec, pre_sec, pre_nsec):
        """ 计算统计时间, 即两个消息时间差"""
        return self._calculate_seconds(now_sec, now_nsec) - self._calculate_seconds(pre_sec, pre_nsec)

    @staticmethod
    def _calculate_seconds(sec, nsec):
        """计算sec + nsec 的和， 单位为seconds"""
        return sec + nsec / 10 ** 9

    @staticmethod
    def _calculate_speed(now_bytes, pre_bytes, delta_time):
        """ 计算统计流量速度"""
        if delta_time:

            return (now_bytes - pre_bytes) / delta_time
        else:
            return 0


    def _calculate_port_speed(self, dpid, port_no, speed):
        curr_bw = speed * 8 / 10 ** 6  # MBit/s
        # print(f"monitorMMMM---> _calculate_port_speed: {curr_bw} MBits/s", )
        self.port_curr_speed.setdefault(dpid, {})
        self.port_curr_speed[dpid][port_no] = curr_bw

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def _port_status_handler(self, ev):
        """处理端口状态：ADD， DELETE, MODIFIED"""
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto

        if msg.reason == ofp.OFPPR_ADD:
            reason = 'ADD'
        elif msg.reason == ofp.OFPPR_DELETE:
            reason = 'DELETE'
        elif msg.reason == ofp.OFPPR_MODIFY:
            reason = 'MODIFY'
        else:
            reason = 'unknown'

        self.logger.info('---> OFPPortStatus received: reason=%s desc=%s',
                         reason, msg.desc)

    # 通过获得的网络拓扑，更新其bw权重
    def create_bandwidth_graph(self):
        link_port_table = self.network_structure.link_port_table
        # print("monitor---->create_bandwidth_graph")
        # print("link_port_table----->", link_port_table)
        for link in link_port_table:
            src_dpid, dst_dpid = link  # 源交换机id,目的交换机id
            # print("src_dpid,dst_dpid",src_dpid,dst_dpid)
            src_port, dst_port = link_port_table[link]
            # print("src_dpid,dst_dpid",src_port, dst_port) #源交换机端口,目的交换机端口

            if src_dpid in self.port_curr_speed.keys() and dst_dpid in self.port_curr_speed.keys():
                src_port_bw = self.port_curr_speed[src_dpid][src_port]
                # print("monitor------> src_port_bw", src_port_bw)
                dst_port_bw = self.port_curr_speed[dst_dpid][dst_port]
                # print("monitor------->dst_port_bw", dst_port_bw)
                src_dst_bandwidth = min(src_port_bw, dst_port_bw)  # 找到链路中最小的带宽
                self.network_structure.graph[src_dpid][dst_dpid]['used_bw'] = src_dst_bandwidth
                # print("monitor------> src_dst_bandwitdh", src_dst_bandwitdh)

                # 对图的edge设置bw值
                capacity = self.network_structure.m_graph[src_dpid][dst_dpid]['free_bw']
                # print("capacity------->",capacity)
                # print("src_dst_bandwidth------->",src_dst_bandwidth)
                self.network_structure.graph[src_dpid][dst_dpid]['free_bw'] = max(capacity - src_dst_bandwidth, 0)

                #对图的edge设置距离
                distance = self.network_structure.m_graph[src_dpid][dst_dpid]['distance']
                self.network_structure.graph[src_dpid][dst_dpid]['distance'] = distance

            else:
                # print("monitor----> not in port_free_bandwidth", src_dpid, dst_dpid)
                self.network_structure.graph[src_dpid][dst_dpid]['free_bw'] = 0

        # print("monitor---> ", self.network_structure.graph.edges(data=True))

    # calculate loss tx - rx / tx
    def calculate_loss_of_link(self):
        """
            发端口 和 收端口 ,端口loss
        """
        for link, port in self.network_structure.link_port_table.items():
            src_dpid, dst_dpid = link
            src_port, dst_port = port
            if (src_dpid, src_port) in self.port_stats_table.keys() and \
                    (dst_dpid, dst_port) in self.port_stats_table.keys():
                # {(dpid, port_no): (stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec,
                # stat.duration_nsec, stat.tx_packets, stat.rx_packets)}
                # 1. 顺向  2022/3/11 packets modify--> bytes
                tx = self.port_stats_table[(src_dpid, src_port)][-1][0]  # tx_bytes
                rx = self.port_stats_table[(dst_dpid, dst_port)][-1][1]  # rx_bytes
                loss_ratio = abs(float(tx - rx) / tx) * 100
                self._save_stats(self.port_loss, link, loss_ratio, 5)

                # 计算错包率
                rx_err = self.port_stats_table[(src_dpid, src_port)][-1][2] # rx_err_bytes 
                rx = self.port_stats_table[(dst_dpid, dst_port)][-1][1]  # rx_bytes
                pkt_err = (rx_err/rx) * 100
                self._save_stats(self.port_pkt_err, link, pkt_err, 5)

                # 计算弃包个数
                tx_packets = self.port_stats_table[(src_dpid, src_port)][-1][-2] # tx_packets
                rx_packets = self.port_stats_table[(dst_dpid, dst_port)][-1][-1]  # rx_packets
                pkt_drop = abs(tx_packets - rx_packets)
                self._save_stats(self.port_pkt_drop, link, pkt_drop, 5)
                # print(f"MMMM--->[{link}]({dst_dpid}, {dst_port}) rx: ", rx, "tx: ", tx,
                #       "loss_ratio: ", loss_ratio)

                # 2. 逆项
                tx = self.port_stats_table[(dst_dpid, dst_port)][-1][0]  # tx_bytes
                rx = self.port_stats_table[(src_dpid, src_port)][-1][1]  # rx_bytes
                loss_ratio = abs(float(tx - rx) / tx) * 100
                self._save_stats(self.port_loss, link[::-1], loss_ratio, 5)

                # 计算错包率
                rx_err = self.port_stats_table[(dst_dpid, dst_port)][-1][2] # rx_err_bytes 
                rx = self.port_stats_table[(src_dpid, src_port)][-1][1]  # rx_bytes
                pkt_err = (rx_err/rx) * 100
                self._save_stats(self.port_pkt_err, link, pkt_err, 5)

                # 计算弃包个数
                tx_packets = self.port_stats_table[(dst_dpid, dst_port)][-1][-2] # tx_packets
                rx_packets = self.port_stats_table[(src_dpid, src_port)][-1][-1]  # rx_packets
                pkt_drop = abs(tx_packets - rx_packets)
                self._save_stats(self.port_pkt_drop, link, pkt_drop, 5)

                # print(f"MMMM--->[{link[::-1]}]({dst_dpid}, {dst_port}) rx: ", rx, "tx: ", tx,
                #       "loss_ratio: ", loss_ratio)
            else:
                pass
                # self.logger.info("MMMM--->  calculate_loss_of_link error", )

    # update graph loss
    def update_graph_loss(self):
        """从1 往2 和 从2 往1,取最大作为链路loss """
        for link in self.network_structure.link_port_table:
            src_dpid = link[0]
            dst_dpid = link[1]
            if link in self.port_loss.keys() and link[::-1] in self.port_loss.keys():
                src_loss = self.port_loss[link][-1]  # 1-->2  -1取最新的那个
                dst_loss = self.port_loss[link[::-1]][-1]  # 2-->1
                link_loss = max(src_loss, dst_loss)  # 百分比 max loss between port1 and port2
                self.network_structure.graph[src_dpid][dst_dpid]['loss'] = link_loss

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.network_structure.graph[src_dpid][dst_dpid]['loss'] = 100

            if link in self.port_pkt_err.keys() and link[::-1] in self.port_pkt_err.keys():
                src_pkt_err = self.port_pkt_err[link][-1]  # 1-->2  -1取最新的那个
                dst_pkt_err = self.port_pkt_err[link[::-1]][-1]  # 2-->1
                link_pkt_err = max(src_pkt_err , dst_pkt_err )  # 百分比 max loss between port1 and port2
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_err'] = link_pkt_err

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_err'] = -1
            
            if link in self.port_pkt_drop.keys() and link[::-1] in self.port_pkt_drop.keys():
                src_pkt_drop = self.port_pkt_drop[link][-1]  # 1-->2  -1取最新的那个
                dst_pkt_drop = self.port_pkt_drop[link[::-1]][-1]  # 2-->1
                link_pkt_drop = max(src_pkt_drop, dst_pkt_drop)  # 百分比 max loss between port1 and port2
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_drop'] = link_pkt_drop

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_drop'] = -1

    def create_loss_graph(self):
        """
            在graph中更新边的loss值
        """
        # self.calculate_loss_of_link()
        self.update_graph_loss()

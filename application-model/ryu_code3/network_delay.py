# -*- coding: utf-8 -*-
"""
@File     : network_delay.py
@Date     : 2022-07-22
@Author   : Terry_Li  -- 长风破浪会有时，直挂云帆济沧海。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.topology.switches import LLDPPacket, Switches

import setting
import time


class NetworkDelay(app_manager.RyuApp):
    """测量链路的时延"""
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'switches': Switches}

    def __init__(self, *args, **kwargs):
        super(NetworkDelay, self).__init__(*args, **kwargs)
        self.name = 'detector'  # 给测量时延的类取个名字

        self.network_structure = lookup_service_brick('discovery')  # 实例化获取拓扑信息的类
        self.network_monitor = lookup_service_brick('monitor')  # 实例化流量监控的类
        self.switch_module = lookup_service_brick('switches')  # 实例化switch内部的类，用来获取ap的ip
        # self.switch_module = kwargs['switches']

        self.echo_delay_table = {}  # {dpid: ryu_ofps_delay}
        self.lldp_delay_table = {}  # {src_dpid: {dst_dpid: delay}}
        self.echo_interval = 0.05   # 发包等待时间

        self._detector_thread = hub.spawn(self.scheduler) # 内部模块测试协程

    def scheduler(self):
        """外部调用协程"""
        hub.sleep(20) # 休眠20s
        while True:
            hub.sleep(setting.DELAY_PERIOD)
            self._send_echo_request()
            self.create_delay_graph()
            if setting.PRINT_SHOW:
                self.show_delay_stats()

    # 利用echo发送时间，与接收时间相减
    # 1.发送echo request
    def _send_echo_request(self):
        """发送echo请求"""
        for datapath in list(self.network_monitor.datapaths_table.values()):
            # print("datapath------>", datapath)
            parser = datapath.ofproto_parser
            data = bytes("%.12f" % time.time(), encoding="utf8")
            # print("delay------data------>",data)
            # 获取当前的时间
            echo_req = parser.OFPEchoRequest(datapath, data)
            # print("delay------echo_req------>",echo_req)
            datapath.send_msg(echo_req)

            # 重要！不要同时发送echo请求，因为它几乎同时会生成大量echo回复。
            # 在echo_reply_处理程序中处理echo reply时，会产生大量队列等待延迟。
            # 防止发太快，这边收不到
            hub.sleep(self.echo_interval)

    # 接收echo reply
    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def _echo_reply_handler(self, ev):
        """
     处理echo响应报文,获取控制器到交换机的链路往返时延

           Controller
               |
  echo latency |
              `|
                Switch
        """
        now_timestamp = time.time()
        data = ev.msg.data
        ryu_ofps_delay = now_timestamp - eval(data)  # 现在的时间减去发送的时间
        # ----------------------------更改dpid---------------------------------#
        dpid = self.network_structure.apid_dict.get(ev.msg.datapath.id)  # 交换机的id标识
        # print("delay------ryu_ofps_delay------>",ryu_ofps_delay)
        self.echo_delay_table[dpid] = ryu_ofps_delay  # 将echo 时延存储到字典中

    # 利用LLDP时延
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """解析LLDP包，这个处理程序可以接收所有可以接收的数据包"""
        # print("delay----->_packet_in_handler")
        try:
            recv_timestamp = time.time()
            msg = ev.msg
            # ---------------------------------更改dpid-----------------------------------#
            dpid = self.network_structure.apid_dict.get(msg.datapath.id)  # 交换机的id标识
            src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
            
        
            # print("delay----->self.switch_module.ports", self.switch_module.ports)

            # 获得key（Port类实例）和data（PortData类实例）
            for port in self.switch_module.ports.keys():  # 开始获取对应交换机端口的发送时间戳
                if src_dpid == port.dpid and src_port_no == port.port_no:  # 匹配key
                    # 获取满足key条件的values值PortData实例，内部保存了发送LLDP报文时的timestamp信息
                    send_timestamp = self.switch_module.ports[port].timestamp
                    if send_timestamp:
                        delay = recv_timestamp - send_timestamp
                    else:
                        delay = 0
                    #-------------------------- 更改src_dpid----------------------#
                    src_dpid = self.network_structure.apid_dict.get(src_dpid)

                    self.lldp_delay_table.setdefault(src_dpid, {})
                    self.lldp_delay_table[src_dpid][dpid] = delay  # 将时延信息存起来
            # print("delay----->lldp_delay_table----->", self.lldp_delay_table)

        except LLDPPacket.LLDPUnknownFormat as e:
            return

    def create_delay_graph(self):
        # 遍历所有的边
        # print("------->"create delay graph)
        for src, dst in self.network_structure.graph.edges:
            delay = self.calculate_delay(src, dst)
            self.network_structure.graph[src][dst]['delay'] = delay

    def calculate_delay(self, src, dst):
        """
                        ┌------Ryu------┐
                        |               |
        src echo latency|               |dst echo latency
                        |               |
                    SwitchA------------SwitchB
                         --->fwd_delay--->
                         <---reply_delay<---
        """

        fwd_delay = self.lldp_delay_table[src][dst]
        reply_delay = self.lldp_delay_table[dst][src]
        ryu_ofps_src_delay = self.echo_delay_table[src]
        # ryu_ofps_dst_delay = self.echo_delay_table[dst]

        # delay = (fwd_delay + reply_delay - ryu_ofps_src_delay - ryu_ofps_dst_delay) / 2
        delay = (fwd_delay + reply_delay - ryu_ofps_src_delay*2) / 2
        return max(delay*1000, 0) # 将时延的单位转换成ms

    def show_delay_stats(self):
        """打印时延信息"""
        self.logger.info("==================================== link %s ====================================", self.name)
        self.logger.info("\n----------------------------")
        self.logger.info("src    dst :    delay")
        for src in self.lldp_delay_table.keys():
            for dst in self.lldp_delay_table[src].keys():
                delay = self.lldp_delay_table[src][dst]
                self.logger.info("%s <---> %s : %s", src, dst, delay*1000)

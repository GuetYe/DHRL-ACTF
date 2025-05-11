# -*- coding: utf-8 -*-
"""
@File     : delay_detect.py
@Date     : 2022-07-20 16:06
@Author   : Terry_Li  - 长路慢慢，唯剑相伴。
IDE       : PyCharm
@Mail     : terry.li.dev@foxmail.com
"""

from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick

from ryu.ofproto import ofproto_v1_3

from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER,CONFIG_DISPATCHER,DEAD_DISPATCHER,HANDSHAKE_DISPATCHER #只是表示datapath数据路径的状态
from ryu.controller.handler import set_ev_cls

from ryu.lib import hub
from ryu.lib.packet import packet,ethernet

from ryu.topology.switches import Switches
from ryu.topology.switches import LLDPPacket

import time

ECHO_REQUEST_INTERVAL = 0.05
DELAY_DETECTING_PERIOD = 5

class DelayDetect(app_manager.RyuApp):
    """ 测量链路的时延 """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self,*args,**kwargs):
        super(DelayDetect,self).__init__(*args,**kwargs)
        self.name = "delay"

        self.topology = lookup_service_brick("topology") #注意：我们使用lookup_service_brick加载模块实例时，对于我们自己定义的app,我们需要在类中定义self.name。
        self.switches = lookup_service_brick("switches") #此外，最重要的是：我们启动本模块DelayDetect时，必须同时启动自定义的模块！！！ 比如：ryu-manager ./TopoDetect.py ./DelayDetect.py --verbose --observe-links

        self.dpid2switch = {} #或者直接为{}，也可以。下面_state_change_handler也会添加进去
        self.dpid2echoDelay = {}

        self.src_sport_dst2Delay = {} #记录LLDP报文测量的时延。实际上可以直接更新，这里单独记录，为了单独展示 {”src_dpid-srt_port-dst_dpid“：delay}

        # self.detector_thread = hub.spawn(self._detector) # 启用时延的线程

    def _detector(self):
        """
        协程实现伪并发，探测链路时延
        """
        while True:
            if self.topology == None:
                self.topology = lookup_service_brick("topology")
            if self.topology.net_flag:
                #print("------------------_detector------------------")
                self._send_echo_request()
                self.get_link_delay()
                if self.topology.net_flag:
                    try:
                        self.show_delay()
                        self.topology.show_topology()
                    except Exception as err:
                        print("------------------Detect delay failure!!!------------------")
            hub.sleep(DELAY_DETECTING_PERIOD) #5秒一次

    def get_link_delay(self):
        """
        更新图中的权值信息
        """
        #print("--------------get_link_delay-----------")
        for src_sport_dst in self.src_sport_dst2Delay.keys():
                src,sport,dst = tuple(map(eval,src_sport_dst.split("-")))
                if src in self.dpid2echoDelay.keys() and dst in self.dpid2echoDelay.keys():
                    sid,did = self.topology.dpid2id[src],self.topology.dpid2id[dst]
                    if self.topology.net_topo[sid][did] != 0:
                        if self.topology.net_topo[sid][did][0] == sport:
                            s_d_delay = self.src_sport_dst2Delay[src_sport_dst]-(self.dpid2echoDelay[src]+self.dpid2echoDelay[dst])/2;
                            if s_d_delay < 0: #注意：可能出现单向计算时延导致最后小于0，这是不允许的。则不进行更新，使用上一次原始值
                                continue
                            self.topology.net_topo[sid][did][1] = self.src_sport_dst2Delay[src_sport_dst]-(self.dpid2echoDelay[src]+self.dpid2echoDelay[dst])/2

    def _send_echo_request(self):
        """
        发生echo报文到datapath
        """
        #print("==========_send_echo_request==============")
        #print(self.dpid2switch)
        for datapath in self.dpid2switch.values():
            parser = datapath.ofproto_parser
            echo_req = parser.OFPEchoRequest(datapath,data=bytes("%.12f"%time.time(),encoding="utf8")) #获取当前时间
            #print("==========_send_echo_request=========2=====")
            datapath.send_msg(echo_req)

            #重要！不要同时发送echo请求，因为它几乎同时会生成大量echo回复。
            #在echo_reply_处理程序中处理echo reply时，会产生大量队列等待延迟。
            hub.sleep(ECHO_REQUEST_INTERVAL)

    @set_ev_cls(ofp_event.EventOFPEchoReply,[MAIN_DISPATCHER,CONFIG_DISPATCHER,HANDSHAKE_DISPATCHER])
    def echo_reply_handler(self,ev):
        """
        处理echo响应报文,获取控制器到交换机的链路往返时延

              Controller
                  |    
     echo latency |  
                 `| 
                   Switch        
        """
        #print("================================")
        #print(ev)
        #print("================================")
        now_timestamp = time.time()
        try:
            echo_delay = now_timestamp - eval(ev.msg.data)
            self.dpid2echoDelay[ev.msg.datapath.id] = echo_delay
        except:
            return


    @set_ev_cls(ofp_event.EventOFPPacketIn,MAIN_DISPATCHER)
    def packet_in_handler(self,ev): #处理到达的LLDP报文，从而获得LLDP时延
        """
                      Controller
                    |        /|\    
                   \|/         |
                Switch----->Switch
        """
        msg = ev.msg
        try:
            src_dpid,src_outport = LLDPPacket.lldp_parse(msg.data) #获取两个相邻交换机的源交换机dpid和port_no(与目的交换机相连的端口)
            dst_dpid = msg.datapath.id #获取目的交换机（第二个），因为来到控制器的消息是由第二个（目的）交换机上传过来的
            dst_inport = msg.match['in_port']
            if self.switches is None:
                self.switches = lookup_service_brick("switches") #获取交换机模块实例

            #获得key（Port类实例）和data（PortData类实例）
            for port in self.switches.ports.keys(): #开始获取对应交换机端口的发送时间戳
                if src_dpid == port.dpid and src_outport == port.port_no: #匹配key
                    port_data = self.switches.ports[port] #获取满足key条件的values值PortData实例，内部保存了发送LLDP报文时的timestamp信息
                    timestamp = port_data.timestamp
                    if timestamp:
                        delay = time.time() - timestamp
                        self._save_delay_data(src=src_dpid,dst=dst_dpid,src_port=src_outport,lldpdealy=delay)
        except:
            return

    def _save_delay_data(self,src,dst,src_port,lldpdealy):
        key = "%s-%s-%s"%(src,src_port,dst)
        self.src_sport_dst2Delay[key] = lldpdealy

    @set_ev_cls(ofp_event.EventOFPStateChange,[MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.dpid2switch:
                self.logger.debug('Register datapath: %016x', datapath.id)
                self.dpid2switch[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.dpid2switch:
                self.logger.debug('Unregister datapath: %016x', datapath.id)
                del self.dpid2switch[datapath.id]

        if self.topology == None:
            self.topology = lookup_service_brick("topology")
        #print("-----------------------_state_change_handler-----------------------")
        #print(self.topology.show_topology())
        #print(self.switches)

    def show_delay(self):
        print("-----------------------show echo delay-----------------------")
        for key,val in self.dpid2echoDelay.items():
            print("s%d----%.12f"%(key,val))
        print("-----------------------show LLDP delay-----------------------")
        for key,val in self.src_sport_dst2Delay.items():
            print("%s----%.12f"%(key,val))
    
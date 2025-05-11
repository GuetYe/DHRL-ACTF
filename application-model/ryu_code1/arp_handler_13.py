# -*- coding: utf-8 -*-
"""
@File     : network_shorest_path.py
@Date     : 2022-07-26
@Author   : Terry_Li  -- 前路漫漫，当克己、当慎独、磨棱角、退优越、沉下心。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, arp, ipv6
from ryu.lib import mac


class ARP_PROXY_13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ARP_PROXY_13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.arp_table = {}
        self.sw = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install table-miss flow entry
        #
        # we specify NO BUFFER to mac_len of the output action due to
        # OVS bug. At this moment, if we specify a lesser number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and truncated packet data.
        # In that case, we cannot output packets correctly.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("switch: %s connected", datapath.id)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                idle_timeout=5, hard_timeout=15,
                                match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)

        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        # self.logger.info("packet in: %s %s %s %s", dpid, src, dst, in_port)

        if pkt.get_protocol(ipv6.ipv6):  # drop the ipv6 packets
            match = parser.OFPMatch(eth_type=eth.ethertype)
            actions = []
            self.add_flow(datapath, 1, match, actions)
            return None

        arp_pkt = pkt.get_protocol(arp.arp)

        if arp_pkt:
            self.arp_table[arp_pkt.src_ip] = src        #ARP learning
            self.logger.info(" ARP: %s -> %s", arp_pkt.src_ip, arp_pkt.dst_ip)
            if self.arp_handler(msg):               #answer or drop
                return None

        self.mac_to_port.setdefault(dpid, {})
        # self.logger.info("packet in: %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        if src not in self.mac_to_port[dpid]:
            self.mac_to_port[dpid][src] = in_port
        # print self.mac_to_port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            # print(self.mac_to_port[dpid])
            out_port = ofproto.OFPP_FLOOD
            # print("Flood")

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time.
        if out_port != ofproto.OFPP_FLOOD:
            # self.logger.info("install flow_mod: %s -> %s", in_port, out_port)

            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def arp_handler(self, msg):
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        arp_pkt = pkt.get_protocol(arp.arp)

        if eth:
            eth_dst = eth.dst
            eth_src = eth.src


        """
        解决环路风暴：
            在回复ARP请求之前，必须解决的是网络环路问题。
            解决方案是：
                以(dpid,eth_src,arp_dst_ip)为key，
                记录第一个数据包的in_port，并将从网络中返回的数据包丢弃，
                保证同一个交换机中的某一个广播数据包只能有一个入口，
                从而防止成环。在此应用中，默认网络中发起通信的第一个数据包都是ARP数据包。
        """
        # sw[(datapath.id, eth_src, arp_dst_ip)] = in_port

        # Break the loop for avoiding ARP broadcast storm
        if eth_dst == mac.BROADCAST_STR:            # and arp_pkt
            arp_dst_ip = arp_pkt.dst_ip
            arp_src_ip = arp_pkt.src_ip

            if (datapath.id, arp_src_ip, arp_dst_ip) in self.sw:
                # packet come back at different port.
                if self.sw[(datapath.id, arp_src_ip, arp_dst_ip)] != in_port:
                    datapath.send_packet_out(in_port=in_port, actions=[])
                    return True
            else:
                self.sw[(datapath.id, arp_src_ip, arp_dst_ip)] = in_port
                print(self.sw)
                self.mac_to_port.setdefault(datapath.id, {})
                self.mac_to_port[datapath.id][eth_src] = in_port

        """
        ARP回复：
            解决完环路拓扑中存在的广播风暴问题之后，要利用SDN控制器获取网络全局的信息的能力，去代理回复ARP请求，
            从而减少网络中泛洪的ARP请求数据。通过自学习主机ARP记录，在通过查询记录并回复。
        """
        if arp_pkt:

            opcode = arp_pkt.opcode

            if opcode == arp.ARP_REQUEST:
                hwtype = arp_pkt.hwtype
                proto = arp_pkt.proto
                hlen = arp_pkt.hlen
                plen = arp_pkt.plen

                arp_src_ip = arp_pkt.src_ip
                arp_dst_ip = arp_pkt.dst_ip

                if arp_dst_ip in self.arp_table:    # arp reply
                    actions = [parser.OFPActionOutput(in_port)]
                    ARP_Reply = packet.Packet()

                    ARP_Reply.add_protocol(ethernet.ethernet(ethertype=eth.ethertype,
                                                             dst=eth.src,
                                                             src=self.arp_table[arp_dst_ip]))
                    ARP_Reply.add_protocol(arp.arp(opcode=arp.ARP_REPLY,
                                                   src_mac=self.arp_table[arp_dst_ip],
                                                   src_ip=arp_dst_ip,
                                                   dst_mac=eth_src, dst_ip=arp_src_ip))

                    ARP_Reply.serialize()

                    out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath,
                                                               buffer_id=datapath.ofproto.OFP_NO_BUFFER,
                                                               in_port=datapath.ofproto.OFPP_CONTROLLER,
                                                               actions=actions, data=ARP_Reply.data)
                    datapath.send_msg(out)
                    print("ARP Reply")
                    return True
        return False

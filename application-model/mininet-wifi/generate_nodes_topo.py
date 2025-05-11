# -*- coding: utf-8 -*-
"""
@File     : network_shorest_path.py
@Date     : 2022-07-27
@Author   : Terry_Li  -- 穷且益坚，不坠青云之志。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import re
import sys
import threading
import time
import random
from mn_wifi.net import Mininet_wifi
from mininet.node import Node, RemoteController, OVSKernelSwitch
from mn_wifi.topo import Topo
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd, _4address
from mn_wifi.node import OVSKernelAP
from mn_wifi.wmediumdConnector import interference
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel, info
from json import dumps
from requests import put
from mininet.util import quietRun
from utils import *

random.seed(2022)


class Nodes14Topo():
    def __init__(self, graph, ap_location, sta_location):
        super(Nodes14Topo, self).__init__()
        "Create a network."
        self.net = Mininet_wifi(controller=RemoteController)

        self.graph = graph
        self.node_idx = graph.nodes
        self.edges_pairs = graph.edges

        self.random_bw = 30  # Gbps -> M * 10
        self.bw4 = 50  # host -- switch

        self.delay = 1  # ms
        self.loss = 2  # %

        self.host_port = 9
        self.snooper_port = 10
        self.ap_position = ap_location  # ['100,100,0', '50,50,0', '150,50,0']
        self.sta_position = sta_location  # ['100,110,0', '40,50,0', '160,50,0']

    def topology(self, args):

        info("*** Creating nodes\n")
        # 添加AP
        APs = {}
        for ap in self.node_idx:
            APs.setdefault(ap, self.net.addAccessPoint(f'ap{ap}',
                                                       ssid=f"ap{ap}-ssid", mode="g", channel="1",
                                                       position=self.ap_position[ap - 1],
                                                       mac="{}0:00:00:00:00:00".format(ap)))
            print('添加AP:', ap)

        # print(APs[1])

        # 添加sta
        STAs = {}
        for sta in self.node_idx:
            # x = int(self.sta_position[sta-1].split(',')[0])
            # y = int(self.sta_position[sta-1].split(',')[1])
            # STAs.setdefault(sta, self.net.addStation(f'sta{sta}', mac=f'00:00:00:00:00:0{sta + 1}',
            #                                          ip=f'192.168.0.{sta}/24', min_x= x, max_x= x+random.randint(0,30),
            #                                          min_y=y, max_y=y+random.randint(0,30), min_v=2.5, max_v=4.0))
            STAs.setdefault(sta, self.net.addStation(f'sta{sta}', mac=f'00:00:00:00:00:0{sta + 1}',
                                                     ip=f'192.168.0.{sta}/24', position=self.sta_position[sta - 1]))
            print('添加STA:', sta)

            # 添加控制器
        c0 = self.net.addController('c0', controller=RemoteController)

        info("*** Configuring Propagation Model\n")
        self.net.setPropagationModel(model="logDistance", exp=4.5)

        info("*** Configuring nodes\n")
        self.net.configureNodes()


        info("*** Adding Links\n")
        ap_port_dict = generate_ap_port(self.graph)
        links_info = {}
        # 添加链路
        for l in self.edges_pairs:
            port1 = ap_port_dict[l[0]].pop(0) + 1
            port2 = ap_port_dict[l[1]].pop(0) + 1

            _bw = random.randint(10, self.random_bw)
            _d = str(random.randint(1, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)
            _distance = round(cal_ap_distance(self.ap_position[l[0] - 1], self.ap_position[l[1] - 1]), 2)

            self.net.addLink(APs[l[0]], APs[l[1]],
                             bw=_bw, delay=_d, loss=_l)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": _bw, "delay": _d, "loss": _l,
                                      "distance": _distance})

        create_topo_links_info_xml(links_info_xml_path, links_info)

        # AP<--->STA
        for i in self.node_idx:
            self.net.addLink(APs[i], STAs[i], bw=self.bw4)

        # ------------------------------------------ 画图---------------------------------------#
        # info("***draw graph\n")
        # if '-p' not in args:
        #     self.net.plotGraph(max_x=600, max_y=600)  # 画出二维的图像

        #  设置站点为移动的
        # self.net.setMobilityModel(time=0, model='RandomDirection', max_x=600, max_y=600,
        #                 seed=20)

        info("*** Starting network\n")
        self.net.build()  # 建立网络
        print("=====================Wait ryu init ===========================")
        time.sleep(5)  # 做一个延时等待ryu的连接
        c0.start()  # 启动ryu控制器
        for i in self.node_idx:
            time.sleep(0.2)
            APs[i].start([c0])

        # -------------------------流量检测----------------------------#
        for i in self.node_idx:
            APs[i].cmd('iw dev %s-mp2 interface add %s-mon0 type monitor' %
                       (APs[i].name, APs[i].name))
            APs[i].cmd('ifconfig %s-mon0 up' % APs[i].name)

        ifname = 'enp2s0'  # have to be changed to your own iface!
        collector = os.environ.get('COLLECTOR', '127.0.0.1')
        sampling = os.environ.get('SAMPLING', '10')
        polling = os.environ.get('POLLING', '10')
        sflow = 'ovs-vsctl -- --id=@sflow create sflow agent=%s target=%s ' \
                'sampling=%s polling=%s --' % (ifname, collector, sampling, polling)

        for ap in self.net.aps:
            sflow += ' -- set bridge %s sflow=@sflow' % ap
            info(' '.join([ap.name for ap in self.net.aps]))
            quietRun(sflow)

        agent = '127.0.0.1'
        topo = {'nodes': {}, 'links': {}}
        for ap in self.net.aps:
            topo['nodes'][ap.name] = {'agent': agent, 'ports': {}}

        path = '/sys/devices/virtual/mac80211_hwsim/'
        for child in os.listdir(path):
            dir_ = '/sys/devices/virtual/mac80211_hwsim/' + '%s' % child + '/net/'
            for child_ in os.listdir(dir_):
                node = child_[:3]
                if node in topo['nodes']:
                    ifindex = open(dir_ + child_ + '/ifindex').read().split('\n', 1)[0]
                    topo['nodes'][node]['ports'][child_] = {'ifindex': ifindex}

        path = '/sys/devices/virtual/net/'
        for child in os.listdir(path):
            parts = re.match('(^.+)-(.+)', child)
            if parts is None: continue
            if parts.group(1) in topo['nodes']:
                ifindex = open(path + child + '/ifindex').read().split('\n', 1)[0]
                topo['nodes'][parts.group(1)]['ports'][child] = {'ifindex': ifindex}
        for l in self.edges_pairs:
            linkName = '%s-%s' % (APs[l[0]].name, APs[l[1]].name)
            topo['links'][linkName] = {'node1': APs[l[0]].name, 'port1': 'APs[{}]-mp2'.format(l[0]),
                                       'node2': APs[l[1]].name, 'port2': 'APs[{}]-mp2'.format(l[1])}

            topo['links'][linkName] = {'node1': APs[l[0]].name, 'port1': APs[l[0]].wintfs[0].name,
                                       'node2': APs[l[1]].name, 'port2': APs[l[1]].wintfs[0].name}

            put('http://127.0.0.1:8008/topology/json', data=dumps(topo))  # 流量发送处理

        # ------------------------------------发流处理------------------------------#
        info("***send flow\n")
        print("---------------get stations device list-----------------------")
        stations = get_mininet_wifi_device(self.net, self.graph.nodes, device='sta')
        print("================Dumping host connections======================")
        dumpNodeConnections(self.net.stations)

        # ----------------------------------------添加网关--------------------------------#
        # run_ip_add_default(stations)
        # # pingall 测试整个网络的通信情况
        # self.net.pingAll()
        # # 做一个单机测试sta1
        # write_pingall_time(finish_file)

        # ---------------------------------------iperf测试--------------------------------#
        # time.sleep(5)
        # stations[1].cmd('iperf3 -s  -p 1002 -1 &')
        # time.sleep(5)
        # stations[2].cmd('iperf3 -c 192.168.0.1 - u -p 1002 -b 9M -t 30  &')

        # time.sleep(5)
        # stations[1].cmd('iperf3 -s  -p 1003 -1 &')
        # time.sleep(5)
        # stations[3].cmd('iperf3 -c 192.168.0.1 - u -p 1003 -b 8M -t 30  &')

        # time.sleep(5)
        # stations[2].cmd('iperf3 -s  -p 2001 -1 &')
        # time.sleep(5)
        # stations[1].cmd('iperf3 -c 192.168.0.2 - u -p 2001 -b 6M -t 30  &')

        # time.sleep(5)
        # stations[3].cmd('iperf3 -s  -p 3001 -1 &')
        # time.sleep(5)
        # stations[1].cmd('iperf3 -c 192.168.0.3 - u -p 3001 -b 4M -t 30  &')

        # ---------------------------------------iperf正式发流-----------------------------#
        # print("====RUn iperf scripts")
        # # bash./iperfTM/TM-0/Client/client_1.sh
        # # bash./iperfTM/TM-1/Server/server_1.sh
        # t = threading.Thread(target=all_host_run_iperf, args=(stations, iperf_path, finish_file))  # 建立一个子线程进行发流
        # print("*****************Thread iperf start*************************")
        # t.start()  # 启动线程
        # all_host_run_iperf(stations, iperf_path, finish_file)
        info("*** Running CLI\n")
        CLI(self.net)

        info("*** Stopping network\n")
        self.net.stop()


if __name__ == "__main__":
    # 文件路径
    xml_topology_path = r'topologies/topology_node_30.xml'  # 链路拓扑路径
    links_info_xml_path = r'links_info/links_info.xml'
    iperf_path = r"./iperfTM"

    iperf_interval = 0
    finish_file = './finish_time.json'

    graph, nodes_num, edges_num, ap_location, sta_location = parse_xml_topology(xml_topology_path)
    mytopo = Nodes14Topo(graph, ap_location, sta_location)

    setLogLevel('info')
    mytopo.topology(sys.argv)

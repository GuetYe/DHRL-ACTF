import random
import xml.etree.ElementTree as ET
import networkx
import sys
import json

from pathlib import Path
from mn_wifi.topo import Topo
from mn_wifi.net import Mininet_wifi
from mininet.node import RemoteController
from mn_wifi.link import wmediumd
from mn_wifi.cli import CLI
from mininet.log import setLogLevel, info
from mn_wifi.wmediumdConnector import interference

random.seed(2021)


def generate_port(node_idx1, node_idx2):
    """
    生成端口
    """
    if (node_idx2 > 9) and (node_idx1 > 9):
        port = str(node_idx1) + "0" + str(node_idx2)
    else:
        port = str(node_idx1) + "00" + str(node_idx2)

    return int(port)


def generate_switch_port(graph):
    """
    生成交换机端口
    """
    switch_port_dict = {}
    for node in graph.nodes:
        switch_port_dict.setdefault(node, list(range(graph.degree[node])))
    return switch_port_dict


def parse_xml_topology(topology_path):
    """
    从topology.xml中解析出topology
    return: topology graph, networkx.Graph()  拓扑图
            nodes_num,  int   节点数
            edges_num, int    链路数
    """
    tree = ET.parse(topology_path)
    root = tree.getroot()
    topo_element = root.find("topology")
    graph = networkx.Graph()
    for child in topo_element.iter():
        # 解析节点
        if child.tag == 'node':
            node_id = int(child.get('id'))
            graph.add_node(node_id)
        # 解析链路
        elif child.tag == 'link':
            from_node = int(child.find('from').get('node'))
            to_node = int(child.find('to').get('node'))
            graph.add_edge(from_node, to_node)

    nodes_num = len(graph.nodes)
    edges_num = len(graph.edges)

    print('nodes: ', nodes_num, '\n', graph.nodes, '\n',
          'edges: ', edges_num, '\n', graph.edges)
    return graph, nodes_num, edges_num


def create_topo_links_info_xml(path, links_info):
    """
        <links_info>
            <links> (switch1, switch2)
                <ports>(1, 1)</ports>
                <bw>100</bw>
                <delay>5ms</delay>
                <loss>1</loss>
            </links>
        </links_info>
    :param path: 保存路径
    :param links_info: 链路信息字典 {link: {ports, bw, delay, loss}}
    :return: None
    """
    # 根节点
    root = ET.Element('links_info')

    for link, info in links_info.items():
        # 一级子节点 links
        child = ET.SubElement(root, 'links')
        child.text = str(link)

        # 二级子节点 （ports, bw, delay, loss）
        sub_child1 = ET.SubElement(child, 'ports')
        sub_child1.text = str((info['port1'], info['port2']))

        sub_child2 = ET.SubElement(child, 'bw')
        sub_child2.text = str(info['bw'])

        sub_child2 = ET.SubElement(child, 'delay')
        sub_child2.text = str(info['delay'])

        sub_child2 = ET.SubElement(child, 'loss')
        sub_child2.text = str(info['loss'])

    tree = ET.ElementTree(root)
    indent(root)  #调整xml格式
    Path(path).parent.mkdir(exist_ok=True)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    print('saved links info xml.')

def indent(elem, level=0): 
    """
    xml调整格式函数
    """
    i = "\n" + level*" " 
    if len(elem): 
     if not elem.text or not elem.text.strip(): 
      elem.text = i + " " 
     if not elem.tail or not elem.tail.strip(): 
      elem.tail = i 
     for elem in elem: 
      indent(elem, level+1) 
     if not elem.tail or not elem.tail.strip(): 
      elem.tail = i 
    else: 
     if level and (not elem.tail or not elem.tail.strip()): 
      elem.tail = i 


class Nodes14Topo():
    def __init__(self, graph):
        super(Nodes14Topo, self).__init__()
        "Create a network."
        self.net = Mininet_wifi(controller=RemoteController, link=wmediumd,
                                wmediumd_mode=interference, config4addr=True)

        self.graph = graph
        self.node_idx = graph.nodes
        self.edges_pairs = graph.edges

        self.random_bw = 30  # Gbps -> M * 10
        self.bw4 = 50  # host -- switch

        self.delay = 20  # ms
        self.loss = 0  # %

        self.host_port = 9
        self.snooper_port = 10
        self.ap_position = ['100,100,0', '50,50,0', '150,50,0']
        self.sta_position = ['100,110,0', '40,50,0', '160,50,0']

    def topology(self, args):

        info("*** Creating nodes\n")
        #添加AP
        APs = {}
        for ap in self.node_idx:
            APs.setdefault(ap, self.net.addAccessPoint(f'ap{ap}', 
                           ssid=f"ap{ap}-ssid", mode="g", channel="1",
                           position=self.ap_position[ap-1]))
            print('添加AP:', ap)

        # print(APs[1])

        #添加sta
        STAs = {}
        for sta in self.node_idx:
            STAs.setdefault(sta, self.net.addStation(f'sta{sta}', mac=f'00:00:00:00:00:0{sta+1}', 
                                 ip=f'192.168.0.{sta+1}/24', position=self.sta_position[sta-1]))
            print('添加STA:', sta) 

        #添加控制器  
        c0 = self.net.addController('c0')

        info("*** Configuring Propagation Model\n")
        self.net.setPropagationModel(model="logDistance", exp=4.5)

        info("*** Configuring nodes\n")
        self.net.configureNodes()

        info("*** Adding Links\n")
        ap_port_dict = generate_switch_port(self.graph)
        links_info = {}
        # 添加链路
        for l in self.edges_pairs:
            port1 = ap_port_dict[l[0]].pop(0) + 1
            port2 = ap_port_dict[l[1]].pop(0) + 1

            _bw = random.randint(5, self.random_bw)
            _d = str(random.randint(1, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)

            self.net.addLink(APs[l[0]], APs[l[1]], port1=port1, port2=port2, cls=wmediumd,
                         bw=_bw, delay=_d, loss=_l)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": _bw, "delay": _d, "loss": _l})

        create_topo_links_info_xml(links_info_xml_path, links_info)
        
        #AP<--->STA
        for i in self.node_idx:
            self.net.addLink(APs[i], STAs[i], bw=self.bw4)
        
        #画图
        if '-p' not in args:
            self.net.plotGraph(max_x=300, max_y=300)

        info("*** Starting network\n")
        self.net.build()
        c0.start()
        for i in self.node_idx:
            APs[i].start([c0])

        info("*** Running CLI\n")
        CLI(self.net)

        info("*** Stopping network\n")
        self.net.stop()


if __name__ == "__main__":
    xml_topology_path = r'topologies/topology_1.xml'
    links_info_xml_path = r'save_links_info/links_info.xml'

    graph, nodes_num, edges_num = parse_xml_topology(xml_topology_path)
    mytopo = Nodes14Topo(graph)

    setLogLevel('info')
    mytopo.topology(sys.argv)

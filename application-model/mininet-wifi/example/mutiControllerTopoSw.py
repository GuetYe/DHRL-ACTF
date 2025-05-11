#!/usr/bin/python

"""
This example create 7 sub-networks to connect 7 domain controllers.
Each domian network contains at least 5 Aps.
For an easy test, we add 1 stations for one Aps.
So, in our topology, we have at least 35 Aps and 35 stations.
Hope it will work perfectly.
"""
from mininet.net import Mininet
from mininet.node import Controller,RemoteController,OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel,info
from mininet.link import Link, Intf, TCLink
from mininet.topo import Topo
import logging
import os

def multiControllerNet(con_num, sw_num, host_num):
    """
    con_num: 控制器的个数
    sw_num: 交换机的个数
    host_num: 主机的个数
    """
    controller_list = []
    sw_list = []
    host_list = []

    net = Mininet(controller=RemoteController)
    
    print("***creating controller\n")
    for i in range(con_num):
        name = 'c%s' % str(i) # 这个c后面不能有空格，切记
        c = net.addController(name,ip = '127.0.0.1', port = 6653 + i)
        controller_list.append(c)
    
    print("***creating switches\n")
    sw_list = [net.addSwitch('s%d'%n) for n in range(sw_num)]
    host_list = [net.addHost('h%d'%n) for n in range(host_num)]
    
    print("***creating links of host2switch\n")
    for i in range(0, sw_num):
        net.addLink(sw_list[i], host_list[i])
        # net.addLink(sw_list[i], host_list[i*2 + 1])

    print("***creating links of switch2switch\n")
    for i in range(0, sw_num, int(sw_num/con_num)):
        for j in range(int(sw_num/con_num)):
            for k in range(int(sw_num/con_num)):
                if j !=k and j > k :
                    net.addLink(sw_list[i+j], sw_list[i+k])
    
    print("***creating intra links of switch2swicth\n")
    # 0-4 5-9 10-14 15-19 20-24 25-29 30-34
    # domain1 -> others
    # net.addLink(sw_list[4], sw_list[6])
    #net.addLink(sw_list[1], sw_list[2])
    # net.addLink(sw_list[1], sw_list[15])
    # net.addLink(sw_list[1], sw_list[20])

    # domain2 -> others
    # net.addLink(sw_list[6], sw_list[10])
    # net.addLink(sw_list[8], sw_list[12])
    # net.addLink(sw_list[8], sw_list[18])
    # net.addLink(sw_list[7], sw_list[25])

    # # domain3 -> others
    # net.addLink(sw_list[10], sw_list[16])
    # net.addLink(sw_list[12], sw_list[16])
    # net.addLink(sw_list[10], sw_list[21])
    # net.addLink(sw_list[12], sw_list[27])

    # # domain4 -> others
    # net.addLink(sw_list[16], sw_list[21])
    # net.addLink(sw_list[18], sw_list[27])
    # net.addLink(sw_list[18], sw_list[31])
    # net.addLink(sw_list[19], sw_list[34])

    # # domain5 -> othe rs
    # net.addLink(sw_list[21], sw_list[27])
    # net.addLink(sw_list[23], sw_list[31])

    # # domain6 -> others
    # net.addLink(sw_list[25], sw_list[31])
    # net.addLink(sw_list[27], sw_list[32])

    # domain7 has not need to add links
    print ("*** Starting network")
    net.build()
    for c in controller_list:
        c.start()
    
    _No = 0
    for i in range(0, sw_num, int(sw_num/con_num)):
        for j in range(int(sw_num/con_num)):
            sw_list[i+j].start([controller_list[_No]])
        _No += 1

    #print "*** Testing network"
    #net.pingAll()

    print ("*** Running CLI")
    CLI(net)

    print ("*** Stopping network")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info') # for CLI output
    multiControllerNet(con_num=2, sw_num=6, host_num=6)
    






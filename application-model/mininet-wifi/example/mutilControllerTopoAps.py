#!/usr/bin/python

"""
    This example create con_num sub-networks to connect con_num  domain controllers.
    Each domain network contains at least sw_num switches.
    For an easy test, we add sta_num for one AP.
    So, in our topology, we have at least 35 APs and 70 Stations.
    Hope it will work perfectly.
"""

from mn_wifi.net import Mininet_wifi
from mininet.node import Controller, RemoteController, OVSSwitch
from mn_wifi.cli import CLI
from mininet.log import setLogLevel, info
from mn_wifi.link import wmediumd
from mininet.topo import Topo
import sys
import logging
import os


def multiControllerNet(args,con_num, aps_num, sta_num):
    "Create a network from semi-scratch with multiple controllers."
    controller_list = []
    aps_list = []
    station_list = []

    net = Mininet_wifi(controller=RemoteController, link=wmediumd)

    for i in range(con_num):
        name = 'c%s' % str(i)
        c = net.addController(name, ip='127.0.0.1', port=6653 + i)
        controller_list.append(c)
        print("*** Creating %s" % name)

    print("*** Creating Aps")
    for n in range(aps_num):
        x = 30 + n*7
        y = 60 + n*3
        z = n
        aps_list.append(net.addAccessPoint('ap%d' % n, position='{}, {}, {}'.format(x,y,z)))

    print ("*** Creating Station")
    for n in range(sta_num):
        x = 45 + n*2
        y = 80 + n*8
        z = n
        station_list.append(net.addStation('sta%d' % n, position='{}, {}, {}'.format(x,y,z)))

    info("***Configuring Propagation Model")
    net.setPropagationModel(model = "logDistance", exp = 4.5)
    info("*** Configuring nodes\n")

    net.configureNodes()

    info ("*** Creating links of sta2aps.")
    for i in range(0, aps_num):
        net.addLink(aps_list[i], station_list[i])
        # net.addLink(switch_list[i], host_list[i*2+1])

    print ("*** Creating interior links of aps2aps.")
    for i in range(0, aps_num, int(aps_num/con_num)):
        for j in range(int(aps_num/con_num)):
            for k in range(int(aps_num/con_num)):
                if j != k and j > k:
                    net.addLink(aps_list[i+j], aps_list[i+k])

    print ("*** Creating intra links of switch2switch.")

    
    # domain1 -> others
    net.addLink(aps_list[0], aps_list[2])
    net.addLink(aps_list[3], aps_list[5])
    # net.addLink(switch_list[1], switch_list[15])
    # net.addLink(switch_list[1], switch_list[20])

    # domain2 -> others
    # net.addLink(switch_list[4], switch_list[6])
    # net.addLink(switch_list[8], switch_list[12])
    # net.addLink(switch_list[8], switch_list[18])
    # net.addLink(switch_list[7], switch_list[25])

    # domain3 -> others
    # net.addLink(switch_list[10], switch_list[16])
    # net.addLink(switch_list[12], switch_list[16])
    # net.addLink(switch_list[10], switch_list[21])
    # net.addLink(switch_list[12], switch_list[27])

    # domain4 -> others
    # net.addLink(switch_list[16], switch_list[21])
    # net.addLink(switch_list[18], switch_list[27])
    # net.addLink(switch_list[18], switch_list[31])
    # net.addLink(switch_list[19], switch_list[34])

    # domain5 -> others
    # net.addLink(switch_list[21], switch_list[27])
    # net.addLink(switch_list[23], switch_list[31])

    # domain6 -> others
    # net.addLink(switch_list[25], switch_list[31])
    # net.addLink(switch_list[27], switch_list[32])

    #domain7 has not need to add links.

    info("***Drawing the topology")
    if '-p' not in args:
        net.plotGraph(max_x = 200, max_y = 200, max_z = 200)


    print ("*** Starting network")
    net.build()
    for c in controller_list:
        c.start()
    
    _No = 0
    for i in range(0, aps_num, int(aps_num/con_num)):
        for j in range(int(aps_num/con_num)):
            aps_list[i+j].start([controller_list[_No]])
        _No += 1

    #print "*** Testing network"
    #net.pingAll()

    print ("*** Running CLI")
    CLI(net)

    print ("*** Stopping network")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')  # for CLI output
    multiControllerNet(sys.argv, con_num=3, aps_num=6, sta_num=6)
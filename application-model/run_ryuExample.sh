#!/bin/bash


cd ~/LJQ/rerouting-drl/application-model/ryu_code1


echo "where there is a will, there is a  way"


echo '21022303074 lijinqiang'
for i in $(seq 1 1);
	do 
	let port=i+6652
	xterm -title "c$i" -hold -e ryu-manager simple_switch_13.py  arp_handler_13.py --ofp-tcp-listen-port=$port &
	done





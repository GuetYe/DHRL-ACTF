#!/bin/bash


cd ~/LJQ/rerouting-drl/application-model/mininet-wifi

# echo '1' | sudo -S mn --wifi -c
echo "loading nodes topo..."

sudo python3 generate_nodes_topo.py 

# sudo python3 mutilControllerTopoAps.py
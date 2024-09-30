import os
import time
# 从终端输入需要执行的命令

print("please input your cmd:")
stime = time.time()
cmd_str = input().strip()
os.system(cmd_str)
etime = time.time()
print("Time required to run the program:{}".format(etime-stime))
print("run over")
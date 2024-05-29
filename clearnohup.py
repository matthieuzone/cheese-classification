import time
import os
import subprocess

while os.stat("nohup.out").st_size != 0:
    os.system(">nohup.out")
    time.sleep(10000)


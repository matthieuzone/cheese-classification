import time
import os

while os.stat("nohup.out").st_size != 0:
    f = open("nohup.out", "w")
    f.close()
    time.sleep(10000)


import threading
import numpy as np
import matplotlib.pyplot as plt
import time

arr = np.array([1, 2, 3, 4, 5])

def iterate():
    global arr
    count=0
    while True:
        arr =np.mod(arr * 12345,4567)
        count+=1
        time.sleep(0.1)
        print("iter:"+str(count))

def output():
    global arr
    count=0

    while True:
        plt.clf()
        plt.plot(arr)
        plt.pause(0.1)
        count+=1
        print("plot:"+str(count))


thread1 = threading.Thread(target=iterate)
# thread2 = threading.Thread(target=output)
thread1.start()
output()

# thread2.start()

# 通过手动停止主线程的方式停止循环
while True:
    pass
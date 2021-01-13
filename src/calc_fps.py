# 2020/10/16 
# https://qiita.com/kzmssk/items/945a3f67311d91c3f0c2 参考

import time, sys
import queue
import numpy as np

class FPSCalculator(object):
    def __init__(self, length=5):
        self.times = queue.LifoQueue()
        self.length = length
        self.fpsHist = []

    def tick_tack(self):
        if self.times.qsize() < self.length:
            self.times.put(time.time())

        elif self.times.qsize() == self.length:
            begin = self.times.get()
            end = time.time()
            self.times.put(end)
            fps = self.length / (end - begin)

            self.fpsHist.append(fps)

    def getFPS(self):
        return np.average(np.array(self.fpsHist[1:]))
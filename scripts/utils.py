import os
import time

class timer():
    def __init__(self):
        self.laps = []

    def start(self):
        self.tic = time.time()

    def stop(self):
        self.laps.append(time.time() - self.tic)
        return self.laps[-1]

    def sum(self):
        return sum(self.laps)

from collections import deque
from gym.spaces import Box
import itertools
import numpy as np
from scipy.signal import lfilter


class CompetenceQueue():
    def __init__(self, window = 100, maxlen=250):
        self.window = window
        self.C = deque(maxlen=maxlen)
        self.C_avg = deque(maxlen=10)
        self.C_avg.append(0)
        self.CP = 0
        self.init_stat()

    def init_stat(self):
        self.envstep = 0
        # self.trainstep = 0
        # self.trainstepT = 0
        # self.attempt = 0
        # self.tutorsample = 0
        # self.terminal = 0

    def process_ep(self, episode, term):
        self.C.append(term)
        self.envstep += len(episode)
        # self.attempt += 1

    # def process_samples(self, samples):
    #     self.trainstep += 1
    #     self.terminal += np.mean(samples['t'])
    #     self.tutorsample += np.mean(samples['o'])
    #
    # def process_samplesT(self, samples):
    #     self.trainstepT += 1

    def update(self):
        size = len(self.C)
        if size > 2:
            window = min(size, self.window)
            self.C_avg.append(np.mean(list(self.C)[-window:]))
            self.CP = self.C_avg[-1] - self.C_avg[0]

    def get_stats(self):
        dict = {'C': float("{0:.3f}".format(self.C_avg[-1])),
                'CP': float("{0:.3f}".format(self.CP)),
                'envstep': float("{0:.3f}".format(self.envstep))}
        return dict
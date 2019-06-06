import numpy as np

class RingBuffer(object):
    def __init__(self, limit):
        self._storage = []
        self._limit = limit
        self._oldest = 0
        self._next = 0
        self._numsamples = 0

    def append(self, idx):
        if self._next >= len(self._storage):
            self._storage.append(idx)
            self._numsamples += 1
        else:
            self._storage[self._next] = idx
            if self._next == self._oldest and self._numsamples > 0:
                self._oldest = (self._oldest + 1) % self._limit
            else:
                self._numsamples += 1
        self._next = (self._next + 1) % self._limit

    def popOrNot(self, idx):
        if idx == self._storage[self._oldest]:
            self._oldest = (self._oldest + 1) % self._limit
            self._numsamples -= 1

    def sample(self, batch_size):
        res = []
        idxs = [np.random.randint(0, self._numsamples - 1) for _ in range(batch_size)]
        for i in idxs:
            idx = (self._oldest + i) % len(self._storage)
            res.append(self._storage[idx])
        return res

class ReplayBuffer(object):
    def __init__(self, limit, names, N):
        self._storage = []
        self._next_idx = 0
        self._limit = limit
        self._names = names
        self._tutorBuffers = [RingBuffer(1000) for _ in range(N)]

    def __len__(self):
        return len(self._storage)

    def append(self, item):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            if self._storage[self._next_idx]['o']:
                for t in np.where(self._storage[self._next_idx]['u'])[0]:
                    self._tutorBuffers[t].popOrNot(self._next_idx)
            self._storage[self._next_idx] = item
        if item['o']:
            for t in np.where(item['u'])[0]:
                self._tutorBuffers[t].append(self._next_idx)
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batchsize):
        res = None
        if len(self._storage) >= 10000:
            idxs = [np.random.randint(0, len(self._storage) - 1) for _ in range(batchsize)]
            exps = []
            for i in idxs:
                exps.append(self._storage[i])
            res = {name: np.array([exp[name] for exp in exps]) for name in self._names}
        return res

    def sampleT(self, batchsize, t):
        res = None
        if self._tutorBuffers[t]._numsamples >= batchsize:
            idxs = self._tutorBuffers[t].sample(batchsize)
            exps = []
            for i in idxs:
                exps.append(self._storage[i])
            res = {name: np.array([exp[name] for exp in exps]) for name in self._names}
        return res
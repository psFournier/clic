import numpy as np
from gym import Wrapper
from samplers.competenceQueue import CompetenceQueue
from buffers import ReplayBuffer

class PlayroomGM(Wrapper):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.eps = float(args['--eps'])
        self.hindsight = bool(int(args['--her']))
        self.demo_f = [int(f) for f in args['--demo'].split(',')]
        self.features = [int(f) for f in args['--features'].split(',')]

        self.feat = np.array(self.features)
        self.N = self.feat.shape[0]
        vs = np.zeros(shape=(self.N, self.state_dim[0]))
        vs[np.arange(self.N), self.feat] = 1
        self.vs = vs / np.sum(vs, axis=1, keepdims=True)
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])
        self.idx = -1
        self.v = np.zeros(shape=(self.state_dim[0], 1))
        self.g = np.ones(shape=(self.state_dim[0]))
        self.queues = [CompetenceQueue() for _ in range(self.N)]
        self.names = ['s0', 'r0', 'a', 's1', 'r1', 'g', 'v', 'o', 'u', 't']
        self.buffer = ReplayBuffer(limit=int(1e5), names=self.names, N=self.N)

    def reset(self, exp):
        self.idx, self.v = self.sample_v(exp['s0'])
        exp['g'] = self.g
        exp['v'] = self.v
        exp['t'] = 0
        return exp

    def get_r(self, s, g, v):
        terms = np.sum(np.multiply(v, s==g), axis=1, keepdims=True)
        rewards = np.zeros(shape=terms.shape)
        rewards[np.where(terms)] = self.rTerm
        rewards[np.where(1 - terms)] = self.rNotTerm
        return terms.squeeze(), rewards.squeeze()

    def sample_v(self, s):
        remaining_v = [i for i in range(self.N) if s[self.feat[i]] != 1]
        probs = self.get_probs(idxs=remaining_v, eps=self.eps)
        idx = np.random.choice(remaining_v, p=probs)
        v = self.vs[idx]
        return idx, v

    def sampleT(self, batch_size):
        idxs = [i for i in range(self.N) if self.buffer._tutorBuffers[i]._numsamples > batch_size]
        if idxs:
            probs = self.get_probs(idxs=idxs, eps=self.eps)
            t = np.random.choice(idxs, p=probs)
            samples = self.buffer.sampleT(batch_size, t)
        else:
            samples, t = None, None
        return samples, t

    def end_episode(self, episode):
        term = episode[-1]['t'][self.idx]
        self.queues[self.idx].process_ep(episode, term)
        base_util = np.zeros(shape=(self.N,))
        base_util[self.idx] = 1
        self.process_trajectory(episode, base_util=base_util, hindsight=self.hindsight)

    def process_trajectory(self, trajectory, base_util=None, hindsight=True):
        if base_util is None:
            u = np.zeros(shape=(self.N,))
        else:
            u = base_util
        u = np.expand_dims(u, axis=1)
        # mcr = np.zeros(shape=(self.N,))
        for exp in reversed(trajectory):
            u = self.gamma * u
            if hindsight:
                u[np.where(exp['r1'] > exp['r0'])] = 1

            # u_idx = np.where(u != 0)
            # mcr[u_idx] = exp['r1'][u_idx] + self.gamma * mcr[u_idx]
            exp['u'] = u.squeeze()
            # exp['mcr'] = mcr
            if any(u!=0):
                self.buffer.append(exp.copy())

    def get_demo(self, task):
        demo = []
        exp = {}
        exp['s0'] = self.env.reset()
        exp['t'], exp['r0'] = self.get_r(exp['s0'], self.g, self.vs)
        exp['g'] = self.g
        exp['v'] = self.vs[list(self.feat).index(task)]
        while True:
            a, done = self.opt_action(task)
            if done:
                break
            else:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env.step(exp['a'], True)[0]
                exp['t'], exp['r1'] = self.get_r(exp['s1'], self.g, self.vs)
                exp['o'] = 1
                demo.append(exp.copy())
                exp['s0'] = exp['s1']
                exp['r0'] = exp['r1']

        return demo

    def opt_action(self, t):
        return self.env.opt_action(t)

    def get_stats(self):
        stats = {}
        for i, f in enumerate(self.feat):
            self.queues[i].update()
            for key, val in self.queues[i].get_stats().items():
                stats[key + str(f)] = val
            self.queues[i].init_stat()
        return stats

    def get_cps(self):
        return [np.maximum(abs(q.CP + 0.05) - 0.05, 0) for q in self.queues]

    def get_probs(self, idxs, eps):
        cps = self.get_cps()
        vals = [cps[idx] for idx in idxs]
        l = len(vals)
        s = np.sum(vals)
        if s == 0:
            probs = [1 / l] * l
        else:
            probs = [eps / l + (1 - eps) * v / s for v in vals]
        return probs

    @property
    def state_dim(self):
        return 2+self.N,

    @property
    def goal_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 5

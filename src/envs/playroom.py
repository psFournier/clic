import numpy as np
from gym import Env
from random import randint

MAP = [
    "00000100000",
    "00000100000",
    "00000100000",
    "00000100000",
    "00000100000",
    "00000111111",
    "00000100000",
    "01010100000",
    "01100100000",
    "00110100000",
    "00110100000"
    ]

walls = np.asarray(MAP)

class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    TOUCH = 4
    # TAKE = 5

class Obj():
    def __init__(self, env, name, pos, prop, dep, tutor_only=False):
        self.env = env
        self.name = name
        self.x, self.y = pos
        self.prop = prop
        self.dep = dep
        self.env.objects.append(self)
        # self.g = self.gencoordinates()
        self.init()
        self.tutor_only = tutor_only

    # def gencoordinates(self):
    #
    #     while True:
    #         x, y = 0, 0
    #         while (self.ix+x, self.iy+y) in self.env.seenpos:
    #             x, y = 0, 0
    #         self.env.seenpos.add((self.ix+x, self.iy+y))
    #         yield (self.ix+x, self.iy+y)

    def init(self):
        # self.x, self.y = next(self.g)
        self.s = 0

    def touch(self, tutor):
        if self.s == 0 and all([o.s == s for o, s in self.dep]) and (not self.tutor_only or tutor):
            self.s = np.random.choice([0, 1], p=self.prop)

    @property
    def state(self):
        return [self.s]

    @property
    def high(self):
        return [1]

    @property
    def low(self):
        return [0]

    # def init(self):
    #     self.x, self.y = next(self.g)
    #     n = 0
    #     p = [1]
    #     for o, s in self.dep[:-1]:
    #         if o.s == s:
    #             n += 1
    #             p[0] -= 0.1
    #             p.append(0.1)
    #     self.s = np.random.choice(range(n+1), p=p)


class Playroom(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, args):
        self.nR = 13
        self.nC = 13
        self.walls = np.zeros((13, 13))
        for i in range(13):
            if i != 3 and i != 9:
                self.walls[6, i] = 1
        for i in range(7, 13):
            self.walls[i, 6] = 1
        self.tutoronly = [int(f) for f in args['--tutoronly'].split(',')]
        self.initialize()

    def initialize(self):
        # self.seenpos = set()
        self.x, self.y = randint(0, 5), randint(0, self.nC - 1)
        # self.seenpos.add((self.x, self.y))
        self.objects = []

        self.keyDoor1 = Obj(self,
                            name='keyDoor1',
                            pos=(0, 12),
                            prop=[0, 1],
                            dep=[])

        self.door1 = Obj(self,
                         name='door1',
                         pos=(6, 9),
                         prop=[0, 1],
                         dep=[(self.keyDoor1, 1)])

        self.chest1 = Obj(self,
                          name='chest1',
                          pos=(12, 12),
                          prop=[0, 1],
                          dep=[(self.keyDoor1, 1), (self.door1, 1)])

        self.keyDoor2 = Obj(self,
                            name='keyDoor2',
                            pos=(0, 0),
                            prop=[0, 1],
                            dep=[])

        self.door2 = Obj(self,
                         name='door2',
                         pos=(6, 3),
                         prop=[0, 1],
                         dep=[(self.keyDoor2, 1)])

        self.chest2 = Obj(self,
                          name='chest2',
                          pos=(12, 0),
                          prop=[0, 1],
                          dep=[(self.keyDoor2, 1), (self.door2, 1)])

        for i, o in enumerate(self.objects):
            o.tutor_only = (i+2 in self.tutoronly)

        self.initstate = self.state.copy()
        self.lastaction = None

    def check_door(self):
        obj = self.underagent()
        if obj <= 0 or self.objects[obj - 1] not in [self.door1, self.door2] or \
                        self.objects[obj - 1].s == 1:
            return True
        else:
            return False

    def step(self, a, tutor=False):
        env_a = a
        if self.lastaction is not None and np.random.rand() < 0.25:
            env_a = self.lastaction
        self.lastaction = a

        if env_a == Actions.UP:
            if self.y < self.nR - 1 and not self.walls[self.x, self.y + 1] and self.check_door():
                self.y += 1

        elif env_a == Actions.DOWN:
            if self.y > 0 and not self.walls[self.x, self.y - 1] and self.check_door():
                self.y -= 1

        elif env_a == Actions.RIGHT:
            if self.x < self.nC - 1 and not self.walls[self.x + 1, self.y] and self.check_door():
                self.x += 1

        elif env_a == Actions.LEFT:
            if self.x > 0 and not self.walls[self.x - 1, self.y] and self.check_door():
                self.x -= 1

        elif env_a == Actions.TOUCH:
            obj = self.underagent()
            if obj != 0:
                self.objects[obj - 1].touch(tutor)

        return np.array(self.state),

    def underagent(self):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x and obj.y == self.y:
                return i+1
        return 0

    def reset(self):
        self.initialize()
        return np.array(self.state)

    def go(self, x , y):
        dx = x - self.x
        dy = y - self.y
        p = []
        if dx > 0 and not self.walls[self.x+1, self.y]:
            p.append(Actions.RIGHT)
        elif dx < 0 and not self.walls[self.x-1, self.y]:
            p.append(Actions.LEFT)
        if dy > 0 and not self.walls[self.x, self.y+1]:
            p.append(Actions.UP)
        elif dy < 0 and not self.walls[self.x, self.y-1]:
            p.append(Actions.DOWN)

        if p:
            return np.random.choice(p)
        else:
            return None

    # def take(self, i):
    #     obj = self.objects[i]
    #     a = self.go(obj.x, obj.y)
    #     if a is None:
    #         return Actions.TAKE, False
    #     else:
    #         return a, False

    def touch(self, o):
        a = self.go(o.x, o.y)
        if a is None:
            return Actions.TOUCH, False
        else:
            return a, False

    @property
    def high(self):
        res = [self.nC - 1, self.nR - 1]
        for obj in self.objects:
            res += obj.high
        return res

    @property
    def state(self):
        res = [self.x/(self.nC-1), self.y/(self.nR-1)]
        for obj in self.objects:
            res += [(a - c)/(b - c) for a,b,c in zip(obj.state, obj.high, obj.low)]
        return res

    @property
    def low(self):
        res = [0, 0]
        for obj in self.objects:
            res += obj.low
        return res

    def opt_action(self, t):
        obj = self.objects[t-2]
        if obj.s == 1:
            return -1, True
        else:
            for dep, val in obj.dep:
                if dep.s != val:
                    return self.touch(dep)
            return self.touch(obj)

if __name__ == '__main__':
    env = Playroom(args={'--tutoronly': '4,5,6'})
    s = env.reset()
    task = np.random.choice([4])
    while True:
        print(s)
        a, done = env.opt_action(task)
        if done:
            break
        else:
            a = np.expand_dims(a, axis=1)
            s = env.step(a, True)[0]



    # def render(self, mode='human'):
    #     outfile = StringIO() if mode == 'ansi' else sys.stdout
    #
    #     out = self.desc.copy().tolist()
    #     out = [[c.decode('utf-8') for c in line] for line in out]
    #     taxirow, taxicol, passidx = self.decode(self.s)
    #     def ul(x): return "_" if x == " " else x
    #     if passidx < 4:
    #         out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
    #         pi, pj = self.locs[passidx]
    #         out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
    #     else: # passenger in taxi
    #         out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)
    #
    #     # No need to return anything for human
    #     if mode != 'human':
    #         return outfile

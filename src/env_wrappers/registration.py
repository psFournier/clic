""" This whole file is almost identical to the gym.envs.registration file. The only difference is the addition of a wrapper_entry_point attribute to the EnvSpec class. I just dit not want to modify the original gym package."""

from gym import error, logger
from env_wrappers import Base

def load(name):
    import pkg_resources # takes ~400ms to load, so we import it lazily
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result

class EnvSpec(object):
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.
    """

    def __init__(self, id, entry_point=None, trials=100, reward_threshold=None, local_only=False, kwargs=None, nondeterministic=False, tags=None, max_episode_steps=None, max_episode_seconds=None, timestep_limit=None, wrapper_entry_point=None):
        self.id = id
        self.wrapper_entry_point = wrapper_entry_point
        self._entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, args):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self._entry_point is not None:
            cls = load(self._entry_point)
            env = cls(args, **self._kwargs)
        else:
            raise error.Error('Attempting to make deprecated envs {}. (HINT: is there a newer registered version of this envs?)'.format(self.id))

        if self.wrapper_entry_point is not None:
            wrapper_cls = load(self.wrapper_entry_point)
            wrapper = wrapper_cls(env, args)
        else:
            wrapper = Base(env, args)
        # # Make the enviroment aware of which spec it came from.
        # env.unwrapped.spec = self
        return env, wrapper

    def __repr__(self):
        return "EnvSpec({})".format(self.id)

class EnvRegistry(object):
    """Register an envs by ID. IDs remain stable over time and are
    guaranteed to resolve to the same environment dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.env_specs = {}

    def make(self, id, args):
        logger.info('Making new envs: %s', id)
        spec = self.env_specs[id]
        env = spec.make(args)
        return env

    def all(self):
        return self.env_specs.values()

    def register(self, id, **kwargs):
        if id in self.env_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.env_specs[id] = EnvSpec(id, **kwargs)

# Have a global registry
registry = EnvRegistry()

def register(id, **kwargs):
    return registry.register(id, **kwargs)

def make(id, args):
    return registry.make(id, args)
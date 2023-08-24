from gym import utils
from Robot import fetch_env


class Env(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, seed):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/pick_and_place.xml',seed, target_offset=0.0, obj_range=0.15, target_range=0.15, initial_qpos=initial_qpos)
        utils.EzPickle.__init__(self)

import sys
sys.path.insert(0, '/home/antonio/Documents/tfm/2020-tfm-antonio-triguero')

import numpy as np
from gym.spaces import Discrete

from agents.ql import QL
from gym_gazebo.envs.f1.discrete.env import MontmeloLineMultivariate

env = MontmeloLineMultivariate(action_space=Discrete(16), observation_space=Discrete(9), 
                            high=np.array([4, 5]), low=np.array([0, -5]), done_reward=-1)
agent = QL(env, epsilon=0.99999, episilon_discount_factor=0.99999, 
        min_epsilon=0.01, learning_rate=0.95, discount_factor=0.95)
agent.train(total_steps=int(1e6), steps_per_update=1)
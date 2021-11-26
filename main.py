import numpy as np
from gym.spaces import Discrete

from agents.ql import QL
from gym_gazebo.envs.f1.discrete.env import MontmeloLineMultivariate

env = MontmeloLineMultivariate(action_space=Discrete(49), observation_space=Discrete(10), 
                               high=np.array([10, 8]), low=np.array([0, -8]), done_reward=-1)

agent = QL(env, epsilon=0.25, episilon_discount_factor=0.99999, min_epsilon=0.01, 
           learning_rate=0.95, discount_factor=0.95, read_model=True,
           saving_path='q_49x10_max_its.csv')

agent.train(total_steps=int(1e7), steps_per_update=1, steps_per_saving=10000, 
            render=True, iteration_velocity=30, print_agent_info=True)
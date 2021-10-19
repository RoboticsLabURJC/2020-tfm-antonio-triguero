from itertools import product
import random

from gym_gazebo.envs.f1.env import MontmeloLineEnv
from agents.agent import Agent
import numpy as np
from pandas import DataFrame

# TODO: Use herencia and redesign for ql and simpleql
class MontmeloLineQL(MontmeloLineEnv):
    def __init__(self):
        from gym.spaces import Box
        super().__init__(action_space=Box(low=-1, high=1, shape=(2,)),
                         observation_space=Box(low=0, high=255, shape=(int(480 * 0.25), int(640 * 0.25), 3)),
                         args={})

        actions_bines = 7
        bin_size = np.abs(self.action_space.high - self.action_space.low) / actions_bines
        discrete_actions = [np.arange(low, high, size) 
                            for low, high, size in zip(self.action_space.low, self.action_space.high, bin_size)]
        self.actions = np.array(list(product(*discrete_actions)))

        observation_bines = 10
        bin_size = self.observation_space.shape[1] // observation_bines
        self.observations = np.array([bin_size]) * np.arange(observation_bines)

    def observe(self, info):
        def in_range(a, e):
            a1 = a
            a2 = np.append(a[1:], self.observation_space.shape[1])
            return ((a1 <= e) & (e < a2)).nonzero()[0][0]

        done = not any(info['mask'][info['mask'].shape[1] // 2])
        if done or self.last_action[0] == 0:
            return 0, -200, True

        screen_center = self.observation_space.shape[1] / 2
        _, line_columns = info['mask'].nonzero()
        line_center = line_columns[len(line_columns) // 2]
        observation = in_range(self.observations, line_center)

        error = np.abs(line_columns - screen_center) / screen_center
        weights = np.arange(0, len(error))
        reward = np.mean((1 - error) * weights) * self.last_action[0]

        return observation, reward, done

    def step(self, action):
        _, _, _, info = super().step(np.array(self.actions[action]))
        observation, reward, done = self.observe(info)
        return observation, reward, done, info

    def reset(self):
        super().reset()
        positions = [
            (-71, 8.75, 0.004, 0, 0, -1, 0),
			(-71, -31.5, 0.004, 0, 0, -1, 0),
            (90, -31.5, 0.004, 0, 0, -1, -90),
            (105, -3.75, 0.004, 0, 0, -1, -90),
            (105, -3.75, 0.004, 0, 0, -1, 0),
		]
        position = random.choice(positions)
        if position is not None:
            self.car.position(*position)

        _, _, _, info = super().step([-1, 0])
        observation, _, _ = self.observe(info)
        return observation

class QL(Agent):
    def __init__(self, env: MontmeloLineQL, epsilon: float, episilon_discount: float,
                min_epsilon: float):
        if not isinstance(env, MontmeloLineQL):
            raise Exception()
        super().__init__(env)
        self.epsilon = epsilon
        self.episilon_discount = episilon_discount
        self.min_epsilon = min_epsilon
        self.q = np.zeros((len(env.observations), len(env.actions)))

    def train(self, total_steps: int, steps_per_update: int):
        super().train(total_steps, steps_per_update)

    def predict(self, observation: int):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.episilon_discount
        if self.epsilon > random.uniform(0, 1):
            return np.random.randint(0, len(self.env.actions))
        return np.argmax(self.q[observation])

    def update(self, episode: DataFrame):
        for _, row in episode.iterrows():
            observation, action, reward, next_observation, _, _ = row.values
            gamma, beta = 0.95, 0.95
            self.q[observation, action] += gamma * (reward + beta * np.max(self.q[next_observation]) - self.q[observation, action])

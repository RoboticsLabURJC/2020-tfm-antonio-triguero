import random

import numpy as np
from pandas import DataFrame

from gym_gazebo.envs.f1.env import MontmeloLineEnv
from agents.agent import Agent

class MontmeloLineQL(MontmeloLineEnv):
    def __init__(self):
        super().__init__()

        self.linear_velocity = -0.8
        actions_bines = 5
        batch_size = np.abs(self.action_space.high[1] - self.action_space.low[1]) / actions_bines
        self.actions = np.arange(self.action_space.low[1], self.action_space.high[1], batch_size)

        observation_bines = 10
        batch_size = self.observation_space.shape[1] // observation_bines
        self.observations = np.array([batch_size]) * np.arange(observation_bines)

    def observe(self, info):
        def in_range(a, e):
            a1 = a
            a2 = np.append(a[1:], self.observation_space.shape[1])
            return ((a1 <= e) & (e < a2)).nonzero()[0][0]

        done = not any(info['mask'][info['mask'].shape[1] // 2])
        if done:
            return 0, -200, done

        screen_center = self.observation_space.shape[1] / 2
        _, line_columns = info['mask'].nonzero()
        line_center = line_columns[len(line_columns) // 2]
        observation = in_range(self.observations, line_center)

        error = 1 - np.abs(line_columns - screen_center) / screen_center
        weights = line_columns / (np.sum(line_columns) + 0.1)
        reward = np.sum(error * weights)

        return observation, reward, done

    def step(self, action):
        _, _, _, info = super().step(np.array([self.linear_velocity, self.actions[action]]))
        observation, reward, done = self.observe(info)
        return observation, reward, done, info

    def reset(self):
        super().reset()
        positions = [
			(-71, -31.5, 0.004, 0, 0, -1, 0),
            (90, -31.5, 0.004, 0, 0, -1, -90),
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
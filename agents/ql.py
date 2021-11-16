from itertools import product
import random

from gym_gazebo.envs.env import Env
from agents.agent import Agent
import numpy as np
from pandas import DataFrame

class QL(Agent):
    def __init__(self, env: Env, epsilon: float, episilon_discount_factor: float,
                min_epsilon: float, learning_rate: float = 0.95, discount_factor: float = 0.95, 
                saving_path: str = 'q.csv', read_model: bool = False):
        super().__init__(env)
        self.epsilon = epsilon
        self.episilon_discount_factor = episilon_discount_factor
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.saving_path = saving_path
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        if read_model:
            self.q = np.genfromtxt(self.saving_path, delimiter=',')

    def predict(self, observation: int):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.episilon_discount_factor
        if self.epsilon > random.uniform(0, 1):
            return np.random.randint(0, self.env.action_space.n)
        return np.argmax(self.q[observation])

    def update(self, episode: DataFrame):
        for _, row in episode.iterrows():
            observation, action, reward, next_observation, _, _ = row.values
            self.q[observation, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q[next_observation]) - self.q[observation, action])

    def save(self):
        np.savetxt(self.saving_path, self.q, delimiter=",")

    def info(self):
        return {'epsilon': self.epsilon}

from typing import Tuple
import numpy as np
import numpy.typing as npt

from gym_gazebo.envs.f1.env import MontmeloLineEnv
from gym.spaces import Discrete
import cv2 as cv
from gym.error import InvalidAction

class MontmeloLineDiscrete(MontmeloLineEnv):
    def __init__(self, action_space: Discrete, observation_space: Discrete, done_reward: float, args={}):
        super().__init__(args=args)
        self.done_reward = done_reward
        self.action_space = action_space
        self.observation_space = observation_space
        self.observations = np.linspace(-1, 1, self.observation_space.n)

    def __process_observation(self, image: np.array) -> int:
        mask = cv.inRange(image, (41, 41, 218), (42, 42, 219))
        mask = cv.Laplacian(mask, cv.CV_64F)

        # Check if done
        done = not mask[-(mask.shape[0] // 8):].any()

        observation, reward, error = 0, self.done_reward, 0
        if not done:
            screen_center = mask.shape[1] / 2
            error = (screen_center - np.mean(mask.nonzero()[1])) / screen_center
            reward = 1 - np.abs(error)
            observation = np.where((self.observations < error) & (np.append(self.observations[1:], np.inf) >= error))[0][0]

        return observation, reward, done, {'mask': mask, 'image': image}

    def reset(self):
        image = super().reset()
        return self.__process_observation(image)[0]

    def step(self, action: np.array) -> Tuple[int, float, bool, dict]:
        image = super().step(action)[0]
        observation, reward, done, info = self.__process_observation(image)
        return observation, reward, done, info

class MontmeloLineUnivarite(MontmeloLineDiscrete):
    def __init__(self, action_space: Discrete, observation_space: Discrete, 
                low: float, high: float, done_reward: float, 
                linear_velocity: float, args={}):
        if high <= low:
            raise Exception('Invalid high and low values. High must be greater than low value.')
        if linear_velocity <= 0:
            raise Exception('Invalid linear velocity. It must be greater than zero')

        super().__init__(action_space, observation_space, done_reward, args=args)
        self.actions = np.linspace(low, high, self.action_space.n)
        self.linear_velocity = linear_velocity

    def __preprocess_action(self, action: int) -> npt.NDArray[np.float64]:
        return np.array([self.linear_velocity, self.actions[action]]).astype(np.float64)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if not self.action_space.contains(action):
            raise InvalidAction()
            
        # Action preprocessing
        action = self.__preprocess_action(action)
        return super().step(action)

class MontmeloLineMultivariate(MontmeloLineDiscrete):
    def __init__(self, action_space: Discrete, observation_space: Discrete, low: np.array, high: np.array, done_reward: float, args={}):
        if high.shape != (2,) or low.shape != (2,) or np.any(high <= low):
            raise Exception('Invalid high and low arrays. ' 
                            'Low values must be lower than high values'
                            'and size of each one must be (2,)')
        if low[0] < 0:
            raise Exception('Low linear velocity must be greater than zero')

        super().__init__(action_space, observation_space, done_reward, args=args)
        actions = np.linspace(low, high, np.sqrt(self.action_space.n).astype(np.uint))
        self.actions = np.array(np.meshgrid(actions[:, 0], actions[:, 1])).T.reshape(-1,2)
        self.last_action = np.array([0, 0])

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if not self.action_space.contains(action):
            raise InvalidAction()

        action = self.actions[action].astype(np.float64)
        observation, reward, done, info = super().step(action)

        self.last_action = np.array([0, 0])
        if not done:
            velocity_plus = action[0]
            stability_plus = 1 / np.abs(self.last_action[1] - action[1])
            reward *= velocity_plus * stability_plus 
            self.last_action = action

        return observation, reward, done, info 
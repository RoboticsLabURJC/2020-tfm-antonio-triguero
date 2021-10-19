from typing import Tuple
import numpy as np
import numpy.typing as npt

from gym_gazebo.envs.f1.env import MontmeloLineEnv
from gym.spaces import Space, Discrete
import cv2 as cv
from gym.error import InvalidAction

class MontmeloLineDiscrete(MontmeloLineEnv):
    def __init__(self, action_space: Discrete, observation_space: Discrete, done_reward: float, args={}):
        super().__init__(args=args)
        self.done_reward = done_reward
        self.action_space = action_space
        self.observation_space = observation_space
        self.observations = np.linspace(-1, 1, self.observation_space.n)

    def __draw_line(self, drawing: np.array, p1: np.array, p2: np.array) -> np.array:
        v = p2 - p1
        c = (drawing.shape[0] - p1[1]) / v[1]
        p3 = v * c + p1
        if np.any(p3 < 0):
            c = ((0 - p1[0]) / v[0])
            p3 = v * c + p1
        return cv.line(drawing, tuple(p1.astype(np.uint)), 
                        tuple(p3.astype(np.uint)), 255, 1)

    def __skinny_line(self, img: np.array) -> npt.NDArray[np.float64]:
        mask = cv.inRange(img, (41, 41, 218), (42, 42, 219))
        mask = cv.Laplacian(mask, cv.CV_64F)
        # It's an invalid mask if there aren't sufficient points
        if np.count_nonzero(mask) < mask.shape[0] // 2:
            return np.zeros_like(mask)

        rows, cols = mask.nonzero()
        rows_with_more_than_one_column = np.array([len(cols[rows == row]) >= 3 for row in rows])
        rows = rows[rows_with_more_than_one_column]
        cols = cols[rows_with_more_than_one_column]

        row = rows[0]
        row_cols = cols[rows == row]
        p1 = np.array([np.mean([row_cols[0], row_cols[-1]]), row]) # First point of mediatrix line
        row = rows[-1]
        row_cols = cols[rows == row]
        p2 = np.array([np.mean([row_cols[0], row_cols[-1]]), row]) # Second point of mediatrix line
        
        result = self.__draw_line(np.zeros_like(mask), p1, p2)
        return result

    def __process_observation(self, image: np.array) -> int:
        mask = self.__skinny_line(image)

        # Check if done
        done = not mask[-1].any()

        # Observation processing and reward calculation
        observation = 1
        reward = self.done_reward
        if not done:
            screen_center = mask.shape[1] / 2
            _, cols = mask.nonzero()
            error = np.sum(screen_center - cols) / (screen_center * mask.shape[0])
            reward = 1 - error
            observation = np.where((self.observations < error) & (np.append(self.observations[1:], np.inf) > error))[0][0]

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

class MontmeloLineDiscrete(MontmeloLineDiscrete):
    def __init__(self, action_space: Discrete, observation_space: Discrete, low: np.array, high: np.array, done_reward: float, args={}):
        if high.shape != (2,) or low.shape != (2,) or np.any(high <= low):
            raise Exception('Invalid high and low arrays.' 
                            'Low values must be lower than high values'
                            'and size of each one must be (2,)')
        if low[0] < 0:
            raise Exception('Low linear velocity must be greater than zero')

        super().__init__(action_space, observation_space, done_reward, args=args)
        actions = np.linspace(low, high, np.sqrt(self.action_space.n).astype(np.uint))
        self.actions = np.array(np.meshgrid(actions[:, 0], actions[:, 1])).T.reshape(-1,2)

    def __preprocess_action(self, action: int) -> npt.NDArray[np.float64]:
        return self.actions[action].astype(np.float64)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if not self.action_space.contains(action):
            raise InvalidAction()

        action = self.__preprocess_action(action)
        observation, reward, done, info = super().step(action)
        reward = reward * action[0]
        return observation, reward, done, info 
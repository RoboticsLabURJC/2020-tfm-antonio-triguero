import os
import tempfile
from csv import writer
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from pandas import DataFrame
from tqdm import tqdm

class History:
    def __init__(self, path: str, filename: str, overwrite: bool = True):
        self.__full_path = os.path.join(path, filename)
        self.__executor = ThreadPoolExecutor(max_workers=1)

        if overwrite:
            self.__write(['observation', 'action', 'reward', 'next_observation', 'done', 'info'], mode = "w+")

    def __write(self, atts: List[Any], mode: str = "a+"):
        def task():
            with open(self.full_path, mode, newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(atts)

        self.__executor.submit(task)

    def append(self, observation, action, reward, next_observation, done, info):
        self.__write([observation, action, reward, next_observation, done, info])

    def close(self):
        self.__executor.shutdown(wait=True)
        return self.__full_path

class Episode:
    def __init__(self, size: int):
        self.__data = DataFrame(index=range(size), columns=['observation', 'action', 'reward', 'next_observation', 'done', 'info'])
        self.__current_index = 0
        self.__executor = ThreadPoolExecutor(max_workers=1)

    def append(self, observation, action, reward, next_observation, done, info):
        def task():
            self.__data.loc[self.__current_index] = [observation, action, reward, next_observation, done, info]
            self.__current_index += 1

        self.__executor.submit(task)

    def close(self):
        self.__executor.shutdown(wait=True)
        return self.__data

class Agent:
    def __init__(self, env):
        self.env = env

    def train(self, total_steps: int, steps_per_update: int, steps_per_saving: int, render: bool = False):
        history = History(tempfile.mkdtemp(), 'history.csv')

        pbar = tqdm(desc='Steps', total=total_steps)
        observation, done = self.env.reset(), False
        for step in range(0, total_steps, steps_per_update):
            episode = Episode(steps_per_update)
            for _ in range(steps_per_update):
                if done:
                    observation, done = self.env.reset(), False

                action = self.predict(observation)
                if render: 
                    self.env.render()
                next_observation, reward, done, info = self.env.step(action)
                
                episode.append(observation, action, reward, next_observation, done, info)
                history.append(observation, action, reward, next_observation, done, info)

                observation = next_observation
                pbar.update()

                if step % steps_per_saving == 0:
                    self.save()

            self.update(episode.close())

        return history.close()

    def predict(self, observation):
        raise NotImplementedError

    def update(self, episode: DataFrame):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
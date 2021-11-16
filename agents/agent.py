import os
import tempfile
from csv import writer
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from gym_gazebo.envs import Env

from pandas import DataFrame
from tqdm import tqdm
import time

class History:
    def __init__(self, path: str, filename: str, overwrite: bool = True):
        self.__full_path = os.path.join(path, filename)
        self.__executor = ThreadPoolExecutor(max_workers=1)

        if overwrite:
            self.__write(['observation', 'action', 'reward', 'next_observation', 'done', 'info'], mode = "w+")

    def __write(self, atts: List[Any], mode: str = "a+"):
        def task():
            with open(self.__full_path, mode, newline='') as write_obj:
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

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            return 0

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time

    def sleep(self, seconds: float):
        if seconds > 0:
            time.sleep(seconds)

class Agent:
    def __init__(self, env: Env):
        self.env = env

    def __run(self, total_steps: int = int(1e6), steps_per_update: int = None, steps_per_saving: int = 100, 
            render: bool = False, iteration_velocity: int = None, stop_and_go_seconds: int = None, 
            print_env_info: bool = False, print_agent_info: bool = False):
        history = History(tempfile.mkdtemp(), 'history.csv')
        iteration_velocity_timer = Timer()
        pbar = tqdm(desc='Steps', total=total_steps)

        episode_size = total_steps if steps_per_update is None else steps_per_update
        observation, done = self.env.reset(), False
        for step in range(0, total_steps, episode_size):
            episode = Episode(steps_per_update)
            for _ in range(steps_per_update):
                if done:
                    observation, done = self.env.reset(), False

                elapsed_time = iteration_velocity_timer.stop()
                if iteration_velocity is not None:
                    iteration_velocity_timer.sleep(1 / iteration_velocity - elapsed_time)

                action = self.predict(observation)

                iteration_velocity_timer.start()

                if stop_and_go_seconds is not None:
                    self.env.unpause()
                    time.sleep(stop_and_go_seconds)
                    self.env.pause()

                if render: 
                    self.env.render()

                next_observation, reward, done, info = self.env.step(action)
                
                episode.append(observation, action, reward, next_observation, done, info)
                history.append(observation, action, reward, next_observation, done, info)

                observation = next_observation
                if print_env_info:
                    pbar.set_postfix(info)
                if print_agent_info:
                    pbar.set_postfix(self.info())
                pbar.update()

                if step % steps_per_saving == 0:
                    self.save()

            if steps_per_update is not None:
                self.update(episode.close())

        return history.close()

    def train(self, total_steps: int = int(1e6), steps_per_update: int = 100, steps_per_saving: int = 100, 
              render: bool = False, iteration_velocity: int = None, stop_and_go_seconds: int = None,
              print_env_info: bool = False, print_agent_info: bool = False):
        return self.__run(total_steps=total_steps, steps_per_update=steps_per_update, steps_per_saving=steps_per_saving,
                        render=render, iteration_velocity=iteration_velocity, stop_and_go_seconds=stop_and_go_seconds,
                        print_env_info=print_env_info, print_agent_info=print_agent_info)

    def inference(self, total_steps: int = int(1e6), render: bool = False, print_env_info: bool = False, print_agent_info: bool = False):
        return self.__run(total_steps=total_steps, render=render, print_env_info=print_env_info, print_agent_info=print_agent_info)

    def predict(self, observation):
        raise NotImplementedError

    def update(self, episode: DataFrame):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError
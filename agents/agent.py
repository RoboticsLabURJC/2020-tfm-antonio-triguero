import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

from pandas import DataFrame
from tqdm import tqdm

class Agent:
    def __init__(self, env):
        self.env = env
        self.executor = ProcessPoolExecutor(max_workers=1)

    def train(self, total_steps: int, steps_per_update: int, steps_per_saving: int):
        columns=['observation', 'action', 'reward', 
                'next_observation', 'done', 'info']
        history_path = os.path.join(tempfile.mkdtemp(), 'history.csv')
        DataFrame(columns=columns).to_csv(history_path)

        pbar = tqdm(desc='Steps', total=total_steps)
        observation, done = self.env.reset(), False
        for step in range(0, total_steps, steps_per_update):
            episode = DataFrame(index=range(steps_per_update), columns=columns)
            for _ in range(steps_per_update):
                if done:
                    observation, done = self.env.reset(), False

                action = self.predict(observation)
                next_observation, reward, done, info = self.env.step(action)
                
                episode.loc[0] = [observation, action, reward, next_observation, done, info]
                observation = next_observation
                pbar.update()

                if step % steps_per_saving == 0:
                    self.save()

            self.update(episode)
            self.executor.submit(episode.copy().to_csv, history_path, mode='a', header=False)

        print('Saving training data...', end=' ')
        print('DONE!')

    def predict(self, observation):
        raise NotImplementedError

    def update(self, episode: DataFrame):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
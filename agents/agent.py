import os
import tempfile

from pandas import DataFrame
from tqdm import tqdm

# TODO: Parallel CSV writing. Queue of processes?
class Agent:
    def __init__(self, env):
        self.env = env

    def train(self, total_steps: int, steps_per_update: int):
        history_path = os.path.join(tempfile.mkdtemp(), 'history.csv')
        columns=['observation', 'action', 'reward', 
                'next_observation', 'done']
        DataFrame(columns=columns).to_csv(history_path)

        pbar = tqdm(desc='Steps', total=total_steps)

        observation, done = self.env.reset(), False
        for _ in range(0, total_steps, steps_per_update):
            episode = DataFrame(columns=columns)
            for _ in range(steps_per_update):
                if done:
                    observation, done = self.env.reset(), False

                action = self.predict(observation)
                next_observation, reward, done, info = self.env.step(action)
                episode = episode.append({
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'next_observation': next_observation,
                    'done': done,
                    'info': info,
                }, ignore_index=True)

                observation = next_observation
                pbar.update()
            self.update(episode)
            episode.to_csv(history_path, mode='a', header=False)

    def predict(self, observation):
        raise NotImplementedError

    def update(self, episode: DataFrame):
        raise NotImplementedError
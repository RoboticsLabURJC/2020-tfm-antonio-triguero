from collections import defaultdict
import itertools
from gym_gazebo.env import F1ROS
import utils
import numpy as np
from gym import spaces 
import plotting
import matplotlib.pyplot as plt
import pickle
import os


class MontmeloLineDiscrete(F1ROS):
  def __init__(self):
    super().__init__("montmelo_line.launch")

    IMG_WIDTH = 640
    IMG_HEIGTH = 480
    MAX_LINEAR_VEL = 10
    MIN_LINEAR_VEL = 0.5
    MAX_ANGULAR_VEL = 8
    MIN_ANGULAR_VEL = -8
    MAX_SPACE_TO_CENTER = (IMG_WIDTH // 2) * (IMG_HEIGTH // 2)
    MIN_SPACE_TO_CENTER = -MAX_SPACE_TO_CENTER
    N_OBSERVATIONS = 9
    N_ACTIONS = 49 # Must be a number that has integer square root 

    bins = N_OBSERVATIONS + 1
    self.obs_to_values = np.array(np.linspace(MIN_SPACE_TO_CENTER, MAX_SPACE_TO_CENTER, bins))
    self.action_to_values = np.array([
      [e1, e2]
      for e1 in np.linspace(MIN_LINEAR_VEL, MAX_LINEAR_VEL, int(np.sqrt(N_ACTIONS)))
      for e2 in np.linspace(MIN_ANGULAR_VEL, MAX_ANGULAR_VEL, int(np.sqrt(N_ACTIONS)))
    ])

    self.action_space = spaces.Discrete(N_ACTIONS)
    self.observation_space = spaces.Discrete(N_OBSERVATIONS)

  def _get_obs(self, img):
    # Sum of distances of each line point to screen center
    points = np.argwhere(img)
    points = points[np.unique(points[:, 0], return_index=True)[1]]
    
    value = np.sum((img.shape[1] // 2) - points[:, 1])

    # Sum to observation (value binned)
    return np.argwhere([self.obs_to_values[i] <= value < self.obs_to_values[i+1] for i in range(self.observation_space.n - 1)])[0, 0]

  def _get_done(self, img):
    return not np.any(img[img.shape[0] - 1])

  def _get_info(self):
    return {}

  def _get_reward(self, done):
    return -10000 if done else 1

  def reset(self):
    img = super().reset()
    img = utils.skeleton(img)
    return self._get_obs(img), self._get_info()

  def step(self, action):
    value = self.action_to_values[action]
    
    img = super().step(value.tolist())
    img = utils.skeleton(img)

    observation = self._get_obs(img)
    done = self._get_done(img)
    reward = self._get_reward(done)
    info = self._get_info()

    super().render()
    return observation, reward, done, info


def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.
       
    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array 
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):
   
        Action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions
                  
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
   
    return policyFunction

def dd():
      return np.zeros(env.action_space.n)

def qLearning(env, num_episodes, discount_factor = 1.0,
                            alpha = 0.6, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""
       
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).

    Q = defaultdict(dd)
    if os.path.exists("q.pkl"):
      with open("q.pkl","rb") as file:
        Q = pickle.load(file)
   
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))    
       
    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)
       
    # For every episode
    for ith_episode in range(num_episodes):
           
        # Reset the environment and pick the first action
        state, _ = env.reset()
           
        for t in itertools.count():
               
            # get probabilities of all actions from current state
            action_probabilities = policy(state)
   
            # choose action according to 
            # the probability distribution
            action = np.random.choice(np.arange(
                      len(action_probabilities)),
                       p = action_probabilities)
   
            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)
   
            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t
               
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
   
            # done is True if episode terminated   
            if done:
                break
                   
            state = next_state

    with open("q.pkl","wb") as file:
      pickle.dump(Q, file)
    return Q, stats


env = MontmeloLineDiscrete()
observation, info = env.reset()

observations = []
for _ in range(10):
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)

  print(f"observation: {observation}")
  observations.append(observation)

  if done:
      observation, info = env.reset()
env.close()

plt.hist(observations, bins=env.observation_space.n)
plt.show()

# Q, stats = qLearning(env, 1000000)
# print(stats)

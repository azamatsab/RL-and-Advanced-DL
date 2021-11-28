from copy import deepcopy

import numpy as np
from tqdm import tqdm


class Q_learn:
    """ Класс, реализующий double q-learning
    
    """
    def __init__(self, env, alpha, gamma, 
                 eps=0.05, eps_decay=1.0):
        self.env = env
        env = self.env()
        sizes = [space.n for space in env.observation_space]
        sizes += [env.action_space.n, 1]
        env.close()
        self.q_sa = np.random.rand(*sizes)
        self.q_sa2 = np.random.rand(*sizes)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.is_eval = False
        
    def choose_action(self, actions):
        actions = actions.reshape(actions.size)
        action = np.argmax(actions)
        if not self.is_eval and np.random.uniform() < self.eps:
            actions = list(range(actions.size))
            actions.remove(action)
            return np.random.choice(actions)
        return action
    
    def update_q(self, table1, table2, reward, obs, action, n_obs):
        table1[obs][action] = table1[obs][action] + self.alpha * (reward \
                                + self.gamma * np.max(table2[n_obs]) - table1[obs][action])

    def update(self, reward, obs, action, n_obs):
        if np.random.uniform() < 0.5:
            self.update_q(self.q_sa, self.q_sa2, reward, obs, action, n_obs)
        else:
            self.update_q(self.q_sa2, self.q_sa, reward, obs, action, n_obs)
                
    def ace_to_int(self, obs):
        obs = list(obs)
        obs[2] = int(obs[2])
        return tuple(obs)
    
    def step(self, obs, env):
        obs = self.ace_to_int(obs)
        action = self.choose_action(self.q_sa[obs] + self.q_sa2[obs])
        n_obs, reward, done, _ = env.step(action)
        n_obs = self.ace_to_int(n_obs)
        return reward, obs, action, n_obs, done
    
    def play_episode(self, env):
        obs = env.reset()
        done = False
        while not done:
            reward, obs, action, n_obs, done = self.step(obs, env)
            self.update(reward, obs, action, n_obs)
            obs = n_obs
        return reward
        
    def train(self, num_episodes=10 ** 4):
        self.is_eval = False
        env = self.env()
        rewards = []
        for _ in range(num_episodes):
            reward = self.play_episode(env)
            rewards.append(reward)
            self.eps = self.eps_decay * self.eps
        env.close()
        return np.mean(rewards)
    
    def eval(self, num_episodes=10 ** 4):
        self.is_eval = True
        env = self.env()
        rewards = []
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                reward, obs, action, n_obs, done = self.step(obs, env)
                obs = n_obs
            rewards.append(reward)
        env.close()
        return np.mean(rewards)
    
    def load_state_dict(self, states):
        self.q_sa = deepcopy(states[0])
        self.q_sa2 = deepcopy(states[1])

    def get_state_dict(self):
        return (deepcopy(self.q_sa), deepcopy(self.q_sa2))

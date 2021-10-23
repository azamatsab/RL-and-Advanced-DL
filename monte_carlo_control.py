from copy import deepcopy

import numpy as np
from scipy.special import softmax
import gym


POSSIBLE_REWARDS = [-1, 1, -2, 2, 0]

class MC_learn:
    def __init__(self, env, gamma, eps, off_policy=True):
        self.env = env
        env = self.env()
        self.action_space = env.action_space.n
        sizes = [space.n for space in env.observation_space]
        sizes += [self.action_space, 1]
        env.close()
        self.sizes = sizes
        self.q_sa = np.random.rand(*sizes)
        self.policy = np.argmax(self.q_sa, axis=-2)
        
        self.cumulate = np.zeros(sizes)
        self.gamma = gamma
        self.is_eval = False
        self.eps = eps
        self.off_policy = off_policy
        
    def choose_action_eps(self, action):
        actions = list(range(self.action_space))
        probs = np.zeros(self.action_space)
        probs[action] = 1 - self.eps
        for act in actions:
            if act != action:
                probs[act] = self.eps / (self.action_space - 1)
        return np.random.choice(actions, p=probs)
    
    def choose_action_soft(self, actions):
        probs = actions.reshape(actions.size)
        actions = list(range(actions.size))
        return np.random.choice(actions, p=probs)
                
    def ace_to_int(self, obs):
        obs = list(obs)
        obs[2] = int(obs[2])
        return tuple(obs)
    
    def behavior(self):
        behavior = softmax(np.random.uniform(size=self.sizes), axis=-2)
        return behavior
    
    def play_episode(self, env):
        obs = env.reset()
        history = []
        done = False
        final_reward = None
        while not done:
            obs = self.ace_to_int(obs)
            if self.off_policy:
                action = self.choose_action_soft(self.behavior_policy[obs])
            else:
                action = self.choose_action_eps(self.policy[obs][0])
            n_obs, reward, done, _ = env.step(action)
            history.append((obs, action, reward))
            obs = n_obs
        
        final_reward = reward
        returns = 0
        imp_weights = 1
        
        for obs, action, reward in reversed(history):
            returns = self.gamma * returns + reward
            self.cumulate[obs][action] += 1
            alpha = imp_weights / self.cumulate[obs][action]
            self.q_sa[obs][action] += alpha * (returns - self.q_sa[obs][action])
            self.policy[obs] = np.argmax(self.q_sa[obs], axis=-2)
            
            if self.off_policy:
                if action != self.policy[obs][0]:
                    break
                imp_weights = imp_weights / self.behavior_policy[obs][action]
        return final_reward
        
    def train(self, num_episodes=10 ** 4):
        self.behavior_policy = self.behavior()
        self.is_eval = False
        env = self.env()
        rewards = []
        for i_episode in range(num_episodes):
            reward = self.play_episode(env)
            rewards.append(reward)
        env.close()
        return np.mean(rewards)
    
    def eval(self, num_episodes=10 ** 4):
        self.is_eval = True
        env = self.env()
        rewards = []
        for i_episode in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                obs = self.ace_to_int(obs)
                action = self.policy[obs][0]
                n_obs, reward, done, _ = env.step(action)
                obs = n_obs
            assert reward in POSSIBLE_REWARDS, reward
            rewards.append(reward)
        env.close()
        return np.mean(rewards)
    
    def load_state_dict(self, states):
        self.policy = deepcopy(states)

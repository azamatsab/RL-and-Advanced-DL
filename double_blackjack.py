from gym import spaces
from gym.envs.toy_text import BlackjackEnv


class DoubleBJ(BlackjackEnv):
    def __init__(self, natural=False):
        super().__init__(natural=natural)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

    def step(self, action):
        if action == 2:
            obs, reward, done, _ = super().step(1)
            if not done:
                obs, reward, done, _ = super().step(0)
            reward *= 2
        else:
            obs, reward, done, _ = super().step(action)
        return obs, reward, done, _

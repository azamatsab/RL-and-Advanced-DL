from split_blackjack import SplitCountBlackjack


class SplitCountDBJ(SplitCountBlackjack):
    def step(self, action):
        if action == 2:
            obs, reward, done, _ = super().step(1)
            if not done:
                obs, reward, done, _ = super().step(0)
            reward *= 2
        else:
            obs, reward, done, _ = super().step(action)
        return obs, reward, done, _

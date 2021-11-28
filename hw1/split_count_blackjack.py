import gym
from gym import spaces
from gym.envs.toy_text.blackjack import cmp, sum_hand, is_bust, score, usable_ace, is_natural
import numpy as np

from count_blackjack import CountBlackjack

COUNT_TABLE = {1: 0.5, 
                2: 0.5, 
                3: 1, 
                4: 1, 
                5: 1.5, 
                6: 1, 
                7: 0.5, 
                8: 0, 
                9: -0.5, 
                10: -1, 
                11: -1}


class SplitCountBlackjack(CountBlackjack):
    def __init__(self, natural=False):
        super().__init__(natural=natural)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(1000), spaces.Discrete(2))
        )
        self.is_split = False
        self.splitted = False
        self.buffer_reward = 0
        self.player2 = None
        self.swap = False

    def cmp_hands_dealer(self, hand):
        reward = cmp(score(hand), score(self.dealer))
        if self.sab and is_natural(hand) and not is_natural(self.dealer):
            # Player automatically wins. Rules consistent with S&B
            reward = 1.0
        elif (
            not self.sab
            and self.natural
            and is_natural(hand)
            and reward == 1.0
        ):
            # Natural gives extra points, but doesn't autowin. Legacy implementation
            reward = 1.5
        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 3:
            if not self.is_split or self.splitted:
                action = 0
            else:
                self.player2 = [self.player.pop()]
                self.player.append(self.draw_card())
                self.player2.append(self.draw_card())
                self.splitted = True
            done = False
            reward = 0
        elif action == 1:  # hit: add a card to players hand and return
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
                if self.player2 is not None and not self.swap:
                    self.swap = True
                    self.player, self.player2 = self.player2, self.player
                    done = False
                    reward = 0.0
                    self.buffer_reward -= 1.0
                else:
                    reward += self.buffer_reward
            else:
                done = False
                reward = 0.0
        if action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward1 = self.cmp_hands_dealer(self.player)
            if self.buffer_reward == 0.0:
                reward2 = self.buffer_reward
            else:
                reward2 = self.cmp_hands_dealer(self.player2)
            reward = reward1 + reward2
        self.is_split = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.count - int(COUNT_TABLE[self.dealer[1]] * 2), int(self.is_split))

    def reset(self):
        if len(self.deck) <= 15:
            self.reset_deck()
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        self.is_split = self.player[0] == self.player[1]
        self.splitted = False
        self.buffer_reward = 0
        self.player2 = None
        self.swap = False
        return self._get_obs()

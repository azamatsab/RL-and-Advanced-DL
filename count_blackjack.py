import gym
from gym import spaces
from gym.envs.toy_text import BlackjackEnv
from gym.envs.toy_text.blackjack import cmp, sum_hand, is_bust, score, usable_ace, is_natural
import numpy as np


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


class CountBlackjack(BlackjackEnv):
    def __init__(self, natural=False):
        super().__init__(natural=natural)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(20), spaces.Discrete(3))
        )
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4

    def reset_deck(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        np.random.shuffle(self.deck)

    def draw_card(self):
        if len(self.deck) <= 15:
            self.reset_deck()
        return int(self.deck.pop())

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(self.draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def count_cards(self):
        count = 0
        for card in self.player:
            count += COUNT_TABLE[card]
        count += COUNT_TABLE[self.dealer[0]]
        return int(count * 2)

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.count_cards())

    def reset(self):
        self.reset_deck()
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_obs()

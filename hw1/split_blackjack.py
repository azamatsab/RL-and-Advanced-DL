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
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
                                            (spaces.Discrete(32), spaces.Discrete(11), 
                                            spaces.Discrete(2), spaces.Discrete(45), 
                                            spaces.Discrete(2))
                                            )
        self.init_vars()

    def init_vars(self):
        self.can_split = False
        self.splitted = False
        self.reward1 = 0
        self.reward2 = 0
        self.player2 = []
        self.player1_done = False
        self.player2_done = True

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

    def hit(self, player):
        player.append(self.draw_card())
        if is_bust(player):
            done = True
            reward = -1.0
        else:
            done = False
            reward = 0.0
        return reward, done

    def hit_(self, player1, player2, player1_done, player2_done, reward1, reward2):
        reward = 0
        done = False
        if player1_done and player2_done:
            reward = reward1 + reward2
            done = True
        elif not player1_done:
            reward, done = self.hit(player1)
            player1_done = done
            if player1_done:
                reward1 = reward
                if player2_done:
                    reward = reward1 + reward2
                else:
                    done = False
                    reward = 0
        return player1_done, reward1, player2_done, reward2, reward, done

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            if self.player1_done:
                self.player1_done, self.player2_done = self.player2_done, self.player1_done
                self.player, self.player2 = self.player2, self.player
                self.reward1, self.reward2 = self.reward2, self.reward1

            if not self.player1_done:
                self.player1_done, self.reward1, \
                self.player2_done, self.reward2, \
                reward, done = self.hit_(self.player, self.player2, 
                                        self.player1_done, self.player2_done, 
                                        self.reward1, self.reward2)
            else:
                done = True
                reward = self.reward1 + self.reward2

        elif action == 3:
            done = False
            reward = 0
            action = 0
            if self.can_split and not self.splitted:
                self.player2 = [self.player.pop()]
                self.player.append(self.draw_card())
                self.player2.append(self.draw_card())
                self.player2_done = False
                self.splitted = True
                reward = 0
                action = None

        if action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            if not self.player1_done:
                reward1 = self.cmp_hands_dealer(self.player)
            else:
                reward1 = self.reward1
            if self.splitted and not self.player2_done:
                reward2 = self.cmp_hands_dealer(self.player2)
            else:
                reward2 = self.reward2

            reward = reward1 + reward2

        self.can_split = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player),  
                self.count - int(COUNT_TABLE[self.dealer[1]] * 2),
                int(self.can_split))

    def reset(self):
        self.reset_deck()
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        self.init_vars()
        self.can_split = self.player[0] == self.player[1]
        return self._get_obs()

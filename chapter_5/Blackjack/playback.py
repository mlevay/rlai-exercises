import itertools
import numpy as np
import pickle

from .common import pickle, unpickle
from .constants import MIN_CARD_SUM


class Playback():
    class Episode():
        def __init__(self):
            self.actors_k = []
            self.states_k_sum = []
            self.states_k_upcard_value = []
            self.states_k_has_usable_ace = []
            self.actions_k = []
            self.rewards_k_plus_1 = []
            
        def __repr__(self):
            s = []
            for step in range(len(self.actions_k)):
                s.append([
                    self.states_k_sum[step], self.states_k_upcard_value[step], self.states_k_has_usable_ace[step],
                    self.actions_k[step], self.rewards_k_plus_1[step]
                ])
            return s
            
        def __getstate__(self):
            # Copy the object's state from self.__dict__ which contains
            # all our instance attributes. 
            state = self.__dict__.copy()
            # Remove the unneeded entries.
            del state['actors_k']
            return state
            
        def __setstate__(self, state):
            # Restore instance attributes (i.e., filename and lineno).
            self.__dict__.update(state)
            self.actors_k = [True] * len(self.actions_k)
            
        def postprocess(self):
            # re-assign final reward to last player turn
            i = None
            if len(self.actors_k) > 0 and self.actors_k[-1] == False:
                i = len(self.actors_k) - 1
                final_reward = self.rewards_k_plus_1[-1]
                while self.actors_k[i] == False:
                    i = i - 1
                self.rewards_k_plus_1[i] = final_reward
                
            # remove dealer turns
            player_turns = [i != False for i in self.actors_k]
            
            valid_turns = player_turns
            self.actors_k = list(itertools.compress(self.actors_k, valid_turns))
            self.states_k_sum = list(itertools.compress(self.states_k_sum, valid_turns))
            self.states_k_upcard_value = list(itertools.compress(self.states_k_upcard_value, valid_turns))
            self.states_k_has_usable_ace = list(itertools.compress(self.states_k_has_usable_ace, valid_turns))
            self.actions_k = list(itertools.compress(self.actions_k, valid_turns))
            self.rewards_k_plus_1 = list(itertools.compress(self.rewards_k_plus_1, valid_turns))
            
    def __init__(self):
        """Episodes for t=0, t=1, ..., t=T-1"""
        self.episodes = []
        
    def start_episode(self):
        self.episodes.append(Playback.Episode())
        
    def end_episode(self):
        self.episodes[-1].postprocess()
        
    def register_actor(self, is_player: bool):
        self.episodes[-1].actors_k.append(is_player)
        
    def register_state(self, player_sum: int, dealer_upcard_value: int, player_has_usable_ace: bool):
        self.episodes[-1].states_k_sum.append(player_sum)
        self.episodes[-1].states_k_upcard_value.append(dealer_upcard_value)
        self.episodes[-1].states_k_has_usable_ace.append(player_has_usable_ace)
        
    def register_action(self, action_type: int):
        self.episodes[-1].actions_k.append(action_type)
        
    def register_reward(self, reward_value: int):
        self.episodes[-1].rewards_k_plus_1.append(reward_value)
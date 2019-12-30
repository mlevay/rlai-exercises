import itertools
import numpy as np
import pickle

from .common import pickle, unpickle
from .constants import MIN_CURRENT_SUM, PICKLE_FILE_PATH_EPISODES


class Playback():
    class Episode():
        def __init__(self):
            self.actors_k = []
            self.states_k_sum = []
            self.states_k_showing_card_value = []
            self.states_k_has_usable_ace = []
            self.actions_k = []
            self.rewards_k_plus_1 = []
            
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
            
        def preprocess(self):
            # re-assign final reward to last player turn
            i = None
            if len(self.actors_k) > 0 and self.actors_k[-1] == False:
                i = len(self.actors_k) - 1
                final_reward = self.rewards_k_plus_1[-1]
                while self.actors_k[i] == False:
                    i = i - 1
                self.rewards_k_plus_1[i] = final_reward
                
            # remove various steps based on filters:
            # remove first steps until player has at least MIN_CURRENT_SUM cards
            large_enough_turns = [True] * len(self.actors_k)
            if len(self.actors_k) > 0 and self.states_k_sum[0] < MIN_CURRENT_SUM:
                for i in range(len(self.actors_k)):
                    card_sum = self.states_k_sum[i]
                    if card_sum < MIN_CURRENT_SUM:
                        large_enough_turns[i] = False
                    else:
                        break
                
            # remove dealer turns
            player_turns = [i != False for i in self.actors_k]
            
            valid_turns = (np.array(player_turns) & np.array(large_enough_turns)).tolist()
            self.actors_k = list(itertools.compress(self.actors_k, valid_turns))
            self.states_k_sum = list(itertools.compress(self.states_k_sum, valid_turns))
            self.states_k_showing_card_value = list(itertools.compress(self.states_k_showing_card_value, valid_turns))
            self.states_k_has_usable_ace = list(itertools.compress(self.states_k_has_usable_ace, valid_turns))
            self.actions_k = list(itertools.compress(self.actions_k, valid_turns))
            self.rewards_k_plus_1 = list(itertools.compress(self.rewards_k_plus_1, valid_turns))
            
    def __init__(self):
        """Episodes for t=0, t=1, ..., t=T-1"""
        self.episodes = []
        self.pi = np.array([])
        
    def start(self, pi):
        self.pi = pi
    
    def end(self):
        valid_episodes = [len(ep.actors_k) > 0 for ep in self.episodes]
        self.episodes = list(itertools.compress(self.episodes, valid_episodes))
        
        self.save()
        
    def start_episode(self):
        self.episodes.append(Playback.Episode())
        
    def end_episode(self):
        self.episodes[-1].preprocess()
        
    def register_actor(self, is_player: bool):
        self.episodes[-1].actors_k.append(is_player)
        
    def register_state(self, player_sum: int, dealer_showing_card_value: int, player_has_usable_ace: bool):
        self.episodes[-1].states_k_sum.append(player_sum)
        self.episodes[-1].states_k_showing_card_value.append(dealer_showing_card_value)
        self.episodes[-1].states_k_has_usable_ace.append(player_has_usable_ace)
        
    def register_action(self, action_type: int):
        self.episodes[-1].actions_k.append(action_type)
        
    def register_reward(self, reward_value: int):
        self.episodes[-1].rewards_k_plus_1.append(reward_value)
        
    def save(self):
        pickle(PICKLE_FILE_PATH_EPISODES, self.episodes)
    
    def load(self):
        return unpickle(PICKLE_FILE_PATH_EPISODES)
        
playback = Playback()
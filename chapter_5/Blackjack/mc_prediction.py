import numpy as np
import os

from .common import pickle, unpickle
from .constants import GAMMA
from .constants import MAX_CURRENT_SUM, MIN_CURRENT_SUM, PICKLE_FILE_PATH_PI, PICKLE_FILE_PATH_V
from .constants import PLAYER_STICKS_AT
from .playback import Playback, playback


class MonteCarloPrediction():
    def __init__(self):
        self._pi = self.init_pi()
        self._v = self.init_v()
        
    def _get_all_states(self):
        all_sums = list(range(MIN_CURRENT_SUM, MAX_CURRENT_SUM+1))
        all_upcard_values = list(range(1, 11))
        all_ace_states = [False, True]
        all_states = np.array(np.meshgrid(all_sums, all_upcard_values, all_ace_states)).T.reshape(-1, 3)
        return all_states
    
    def init_pi(self):
        if not os.path.exists(PICKLE_FILE_PATH_PI):
            all_states = self._get_all_states()
            
            pi = np.zeros((200,4), dtype=int)
            pi[:,:-1] = all_states
            pi[:,-1] = (pi[:,0] < PLAYER_STICKS_AT).astype(int)

            pickle(PICKLE_FILE_PATH_PI, pi)
            
        return self.load_pi()
        
    def load_pi(self):
        return unpickle(PICKLE_FILE_PATH_PI)
    
    def save_pi(self, pi):
        pickle(PICKLE_FILE_PATH_PI, pi)
    
    def init_v(self):
        if not os.path.exists(PICKLE_FILE_PATH_V):
            all_states = self._get_all_states()
            
            v = np.zeros((200,5), dtype=float)
            v[:,:-2] = all_states.astype(float)

            pickle(PICKLE_FILE_PATH_V, v)
            
        return self.load_v()
        
    def load_v(self):
        return unpickle(PICKLE_FILE_PATH_V)
    
    def save_v(self, v):
        pickle(PICKLE_FILE_PATH_V, v)
    
    def compute(self, episodes: []):
        for episode in episodes:
            G = 0.
            for i in range(len(episode.actors_k) - 1, -1, -1):
                #actors_k = episode.actors_k[i]
                states_k_sum = episode.states_k_sum[i]
                states_k_upcard_value = episode.states_k_upcard_value[i]
                states_k_has_usable_ace = episode.states_k_has_usable_ace[i]
                #actions_k = episode.actions_k[i]
                rewards_k_plus_1 = episode.rewards_k_plus_1[i]
                
                G = GAMMA*G + rewards_k_plus_1
                prev_states_in_episode = [
                    [a, b, c] for a, b, c in zip(
                        episode.states_k_sum[:i], 
                        episode.states_k_upcard_value[:i], 
                        episode.states_k_has_usable_ace[:i])]
                if not ([states_k_sum, states_k_upcard_value, states_k_has_usable_ace] in prev_states_in_episode):
                    index = np.where((self._v[:,0].astype(int) == states_k_sum) & \
                        (self._v[:,1].astype(int) == states_k_upcard_value) & \
                        (self._v[:,2].astype(int) == states_k_has_usable_ace))[0][0]
                    row = self._v[index,:]
                    
                    N = row[4]
                    N += 1
                    row[4] = N
                    V = row[3]
                    row[3] = V + ((G - V)/N)
                    
                    self._v[index,:] = row
        self.save_v(self._v)
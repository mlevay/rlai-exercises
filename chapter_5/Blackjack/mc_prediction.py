import numpy as np
import os

from .common import get_all_states, pickle, unpickle
from .constants import GAMMA
from .constants import DIR_ABS_PATH, DIR_REL_PATH_PRED, PICKLE_FILE_NAME_PRED_V


class MonteCarloPrediction():
    """
    Implements fixed-policy (HIT20) estimation for the state value function using Monte Carlo ES.
    """
    def __init__(self):
        self.file_name_v = self._get_file_path()
        self._v = self._init_v()

    def _get_file_path(self) -> str:
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_PRED)
        return os.path.join(rel_path, PICKLE_FILE_NAME_PRED_V)
    
    def _init_v(self):
        """
        Creates and returns a new state value function (all values = 0.0, all counts = 0). 
        """
        all_states = get_all_states()
            
        v = np.zeros((len(all_states), 5), dtype=float)
        v[:,:-2] = all_states.astype(float)

        self._save_v(v)
        return v
        
    def load_v(self) -> np.ndarray:
        """
        Loads and returns the state value function from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_v)
    
    def _save_v(self, v):
        """
        Saves the state value function to disk.
        """
        pickle(self.file_name_v, v)
    
    def compute_v(self, episodes: []) -> np.ndarray:
        """
        Estimates the value function using Monte Carlo ES and the specified episodes.
        """
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
                    # record a new average value for this state
                    # columns for self._v: 0=sum, 1=upcard, 2=usable ace, 3=state value, 4=number of visits
                    # index for self._v: first 3 columns
                    index = np.where((self._v[:, 0].astype(int) == states_k_sum) & \
                        (self._v[:, 1].astype(int) == states_k_upcard_value) & \
                        (self._v[:, 2].astype(int) == states_k_has_usable_ace))[0][0]
                    row = self._v[index, :]
                    
                    N = row[4]
                    N += 1
                    row[4] = N
                    V = row[3]
                    row[3] = V + ((G - V)/N)
                    
                    self._v[index, :] = row
        self._save_v(self._v)
        return self._v
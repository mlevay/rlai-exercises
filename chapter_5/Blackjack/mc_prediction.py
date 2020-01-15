import numpy as np
import os

from .common import get_all_states, pickle, unpickle
from .constants import GAMMA
from .constants import DIR_ABS_PATH, DIR_REL_PATH_PRED, PICKLE_FILE_NAME_STATS
from .stats import Stats


class MonteCarloPrediction():
    """
    Implements fixed-policy (HIT20) estimation for the state value function using Monte Carlo ES.
    """
    def __init__(self, stats: Stats):
        self.file_name_stats = self._get_file_path()
        self.stats = stats

    def _get_file_path(self) -> str:
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_PRED)
        return os.path.join(rel_path, PICKLE_FILE_NAME_STATS)
        
    def load_stats(self) -> np.ndarray:
        """
        Loads and returns the stats from disk, or None if they doesn't exist.
        """
        return unpickle(self.file_name_stats)
    
    def save_stats(self, stats: np.ndarray):
        """
        Saves the stats to disk.
        """
        pickle(self.file_name_stats, stats)
    
    def compute_v(self, episodes: []) -> np.ndarray:
        """
        Estimates the value function using Monte Carlo ES and the specified episodes.
        """
        for ep in episodes:
            G = 0.
            for i in range(len(ep.actors_k) - 1, -1, -1):
                cs, uc, hua = \
                    ep.states_k_sum[i], ep.states_k_upcard_value[i], ep.states_k_has_usable_ace[i]
                r = ep.rewards_k_plus_1[i]
                
                G = GAMMA*G + r
                prev_states_in_ep = [[a, b, c] for a, b, c in zip(
                        ep.states_k_sum[:i], ep.states_k_upcard_value[:i], ep.states_k_has_usable_ace[:i])]
                if not ([cs, uc, hua] in prev_states_in_ep):
                    # record a new average value for this state
                    self.stats.increment_visit_count(cs, uc, hua)
                    V = self.stats.get_v(cs, uc, hua)
                    N = self.stats.get_visit_count(cs, uc, hua)
                    V = V + ((G - V)/N)
                    self.stats.set_v(cs, uc, hua, V)
        stats = self.stats.get_stats()
        self.save_stats(stats)
        return stats
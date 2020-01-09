import numpy as np
import os
from sklearn.utils.extmath import softmax

from .common import get_all_states_and_actions, pickle, unpickle
from .constants import GAMMA
from .constants import DIR_ABS_PATH, DIR_REL_PATH_CTRL, PICKLE_FILE_NAME_CTRL_PI, PICKLE_FILE_NAME_CTRL_Q


class MonteCarloControl():
    """
    Implements estimation for the optimal action value function using Monte Carlo ES.
    """
    def __init__(self):
        self.file_name_pi, self.file_name_q = self._get_file_paths()
        self._pi = None
        self._q = self.init_q()

    def _get_file_paths(self) -> (str, str):
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_CTRL)
        return os.path.join(rel_path, PICKLE_FILE_NAME_CTRL_PI), os.path.join(rel_path, PICKLE_FILE_NAME_CTRL_Q)   
        
    def load_pi(self) -> np.ndarray:
        """
        Loads and returns the policy function from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_pi)
    
    def save_pi(self, pi):
        """
        Saves the policy function to disk.
        """
        pickle(self.file_name_pi, pi)
    
    def init_q(self):
        """
        Creates and returns a new action value function (all values = 0.0, all counts = 0). 
        """
        all_states_and_actions = get_all_states_and_actions()
        
        q = np.zeros((len(all_states_and_actions), 6), dtype=float)
        q[:,:-2] = all_states_and_actions.astype(float)

        self.save_q(q)            
        return q
        
    def load_q(self) -> np.ndarray:
        """
        Loads and returns the action value function from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_q)
    
    def save_q(self, q):
        """
        Saves the action value function to disk.
        """
        pickle(self.file_name_q, q)
    
    def compute(self, episodes: [], pi: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Estimates the optimal policy and action value function using Monte Carlo ES and the specified initial policy and episodes.
        """
        self._pi = pi
        
        for episode in episodes:
            G = 0.
            for i in range(len(episode.actors_k) - 1, -1, -1):
                #actors_k = episode.actors_k[i]
                states_k_sum = episode.states_k_sum[i]
                states_k_upcard_value = episode.states_k_upcard_value[i]
                states_k_has_usable_ace = episode.states_k_has_usable_ace[i]
                actions_k = episode.actions_k[i]
                rewards_k_plus_1 = episode.rewards_k_plus_1[i]
                
                G = GAMMA*G + rewards_k_plus_1
                prev_states_and_actions_in_episode = [
                    [a, b, c, d] for a, b, c, d in zip(
                        episode.states_k_sum[:i], 
                        episode.states_k_upcard_value[:i], 
                        episode.states_k_has_usable_ace[:i],
                        episode.actions_k[:i])]
                if not ([states_k_sum, 
                         states_k_upcard_value, 
                         states_k_has_usable_ace, 
                         actions_k] in prev_states_and_actions_in_episode):
                    # record a new average value for this state and action
                    # columns for self._q: {0=sum, 1=upcard, 2=usable ace, 3=action}, 4=action value, 5=number of visits
                    q_index = np.where((self._q[:, 0].astype(int) == states_k_sum) & \
                        (self._q[:, 1].astype(int) == states_k_upcard_value) & \
                        (self._q[:, 2].astype(int) == states_k_has_usable_ace) & \
                        (self._q[:, 3].astype(int) == actions_k))[0][0]
                    q_row = self._q[q_index, :]
                    
                    N = q_row[5]
                    N += 1
                    q_row[5] = N
                    Q = q_row[4]
                    q_row[4] = Q + ((G - Q)/N)
                    
                    # revise the policy for this state
                    q_indices = list(np.where((self._q[:, 0].astype(int) == states_k_sum) & \
                        (self._q[:, 1].astype(int) == states_k_upcard_value) & \
                        (self._q[:, 2].astype(int) == states_k_has_usable_ace)))[0].tolist()
                    assert len(q_indices) == 2
                    q_rows = np.array([self._q[ind, :] for ind in q_indices]) 
                    ind_max_q = np.argmax(q_rows[:, 4], axis=0)
                    maximizing_a = int(q_rows[ind_max_q, 3])
                    # columns for self._pi: {0=sum, 1=upcard, 2=usable ace}, 3=action
                    pi_index = np.where((self._pi[:, 0].astype(int) == states_k_sum) & \
                        (self._pi[:, 1].astype(int) == states_k_upcard_value) & \
                        (self._pi[:, 2].astype(int) == states_k_has_usable_ace))[0][0]
                    pi_row = self._pi[pi_index, :]
                    pi_row[3] = maximizing_a
        self.save_pi(self._pi)
        self.save_q(self._q)
        return self._pi, self._q
    
    def compute_v_from_q(self, v_init, q):
        # V(s) = sum_over_a[pi(a|s)*Q(s,a)] = sum_over_a[.5*Q(s,a)]
        # columns for v: {0=sum, 1=upcard, 2=usable ace}, 3=state value, 4=number of visits
        # columns for q: {0=sum, 1=upcard, 2=usable ace, 3=action, 4=action value}, 5=number of visits
        index = 0
        for v_row in v_init:
            s_card_sum, s_dealer_upcard, s_usable_ace = v_row[0], v_row[1], v_row[2]
            s_qs = q[
                (q[:, 0] == s_card_sum) & (q[:, 1] == s_dealer_upcard) & (q[:, 2] == s_usable_ace), :
            ]
            assert s_qs.shape[0] == 2
            v_init[index, 3:] = [.5*s_qs[0, 4] + .5*s_qs[1, 4], s_qs[0, 5] + s_qs[1, 5]]
            index += 1
        return v_init

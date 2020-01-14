from abc import ABCMeta, abstractmethod
import numpy as np
import os
import random

from .common import get_all_states_and_actions, pickle, unpickle
from .constants import EPSILON, GAMMA
from .constants import DIR_ABS_PATH, DIR_REL_PATH_CTRL
from .constants import PICKLE_FILE_NAME_CTRL_PI, PICKLE_FILE_NAME_CTRL_Q, PICKLE_FILE_NAME_CTRL_V
from .playback import Playback
from .stats import MCControlESStats, MCControlOnPolicyStats, Stats

class MonteCarloControl(object, metaclass=ABCMeta):
    """
    Implements common functionality for Monte Carlo Control methods.
    """
    def __init__(self, stats: Stats):
        self.file_name_pi, self.file_name_q, self.file_name_v = self._get_file_paths()
        self.stats = stats
        self._pi, self._q, self._v = None, None, None

    def _get_file_paths(self) -> (str, str):
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_CTRL)
        pi_path = os.path.join(rel_path, PICKLE_FILE_NAME_CTRL_PI)
        q_path = os.path.join(rel_path, PICKLE_FILE_NAME_CTRL_Q)  
        v_path = os.path.join(rel_path, PICKLE_FILE_NAME_CTRL_V)  
        return pi_path, q_path, v_path
        
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
        
    def load_v(self) -> np.ndarray:
        """
        Loads and returns the state value function from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_v)
    
    def save_v(self, v):
        """
        Saves the state value function to disk.
        """
        pickle(self.file_name_v, v)
        
    def start_compute(self):
        pass
    
    def end_compute(self):
        self.save_pi(self._pi)
        self.save_q(self._q)
    
class MonteCarloControl_ES_FirstVisit(MonteCarloControl):
    """
    Implements estimation for the optimal action value function using Monte Carlo 
    Control ES (first-visit).
    """
    def __init__(self, stats: MCControlESStats):
        super(MonteCarloControl_ES_FirstVisit, self).__init__(stats)
    
    def compute_v_from_q(self, v_init, q) -> np.ndarray:
        # V(s) = sum_over_a[pi(a|s)*Q(s,a)] = sum_over_a[.5*Q(s,a)]
        stats = self.stats.get_stats()
        for i in list(range(stats.shape[0])):
            row = stats[i] # (len(cols))
            s_card_sum, s_dealer_upcard, s_usable_ace = \
                row[Stats.COL_CARD_SUM], row[Stats.COL_UPCARD], row[Stats.COL_HAS_USABLE_ACE]
            row_stick = self.stats.get_stats(s_card_sum, s_dealer_upcard, s_usable_ace, action=0) # (len(cols))
            row_hit = self.stats.get_stats(s_card_sum, s_dealer_upcard, s_usable_ace, action=1) # (len(cols))
            
            total_visits = row_stick[MCControlESStats.COL_VISITS] + row_hit[MCControlESStats.COL_VISITS]
            prob_stick = row_stick[MCControlESStats.COL_VISITS] / total_visits
            prob_hit = s_srow_hit[MCControlESStats.COL_VISITS] / total_visits
            v = prob_stick*row_stick[MCControlESStats.COL_Q_OF_S_A] + prob_hit*row_hit[MCControlESStats.COL_Q_OF_S_A]
            
            self.stats.set_v_and_probs(s_card_sum, s_dealer_upcard, s_usable_ace, prob_stick, prob_hit, v)
            
        self._v = self.stats.get_stats()[:, [
            Stats.COL_CARD_SUM, Stats.COL_UPCARD, Stats.COL_HAS_USABLE_ACE, 
            MCControlESStats.COL_V_OF_S, MCControlESStats.COL_VISITS]]
        return self._v
            
    def compute_episode(self, ep: Playback.Episode, pi: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Estimates the optimal policy and action value function using Monte Carlo ES 
        and the specified initial policy and episodes.
        """
        self._pi = pi
        
        G = 0.
        for i in range(len(ep.actors_k) - 1, -1, -1):
            cs, uc, hua, a, r = \
                ep.states_k_sum[i], ep.states_k_upcard_value[i], ep.states_k_has_usable_ace[i], \
                ep.actions_k[i], ep.rewards_k_plus_1[i]
            
            G = GAMMA*G + r
            prev_states_and_actions_in_ep = [[a, b, c, d] for a, b, c, d in zip(
                ep.states_k_sum[:i], ep.states_k_upcard_value[:i], ep.states_k_has_usable_ace[:i], ep.actions_k[:i])]
            if not ([cs, uc, hua, a] in prev_states_and_actions_in_ep):
                # record a new average value for this state and action
                self.stats.increment_visit_count(cs, uc, hua, a)
                Q = self.stats.get_q(cs, uc, hua, a)
                N = self.stats.get_visit_count(cs, uc, hua, a)
                Q = Q + ((G - Q)/N)
                self.stats.set_q(cs, uc, hua, a, Q)
                
                # revise the policy for this state (use MC with a deterministic policy)
                row_stick = self.stats.get_stats(cs, uc, hua, action=0) # (len(cols))
                row_hit = self.stats.get_stats(cs, uc, hua, action=1) # (len(cols))
                if row_stick[MCControlESStats.COL_Q_OF_S_A] >= row_hit[MCControlESStats.COL_Q_OF_S_A]:
                    maximizing_a = int(row_stick[MCControlESStats.COL_A])
                else:
                    maximizing_a = int(row_hit[MCControlESStats.COL_A])
                self.stats.set_pi(cs, uc, hua, maximizing_a)
        
        self._pi, self._q = self.stats.get_pis(), self.stats.get_qs()
        return self._pi, self._q
    
class MonteCarloControl_OnP_FirstVisit(MonteCarloControl):
    """
    Implements estimation for the optimal action value function using On-Policy 
    Monte Carlo Control (first-visit).
    """
    def __init__(self, stats: MCControlOnPolicyStats):
        super(MonteCarloControl_OnP_FirstVisit, self).__init__(stats)
    
    def compute_episode(self, ep: Playback.Episode, pi: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Estimates the optimal policy and action value function using Monte Carlo ES 
        and the specified initial policy and episodes.
        """
        self._pi = pi
        
        G = 0.
        for i in range(len(ep.actors_k) - 1, -1, -1):
            #actors_k = ep.actors_k[i]
            cs = ep.states_k_sum[i]
            uc = ep.states_k_upcard_value[i]
            hua = ep.states_k_has_usable_ace[i]
            a = ep.actions_k[i]
            r = ep.rewards_k_plus_1[i]
            
            G = GAMMA*G + r
            prev_states_and_actions_in_episode = [
                [a, b, c, d] for a, b, c, d in zip(
                    ep.states_k_sum[:i], 
                    ep.states_k_upcard_value[:i], 
                    ep.states_k_has_usable_ace[:i],
                    ep.actions_k[:i])]
            if not ([cs, 
                    uc, 
                    hua, 
                    a] in prev_states_and_actions_in_episode):
                # record a new average value for this state and action
                # columns for self._q: {0=sum, 1=upcard, 2=usable ace, 3=action}, 4=q, 5=visit count
                q_index = np.where((self._q[:, 0].astype(int) == cs) & \
                    (self._q[:, 1].astype(int) == uc) & \
                    (self._q[:, 2].astype(int) == hua) & \
                    (self._q[:, 3].astype(int) == a))[0][0]
                q_row = self._q[q_index, :]
                
                N = q_row[5]
                N += 1
                q_row[5] = N
                Q = q_row[4]
                q_row[4] = Q + ((G - Q)/N)
                
                # revise the policy for this state
                q_indices = list(np.where((self._q[:, 0].astype(int) == cs) & \
                    (self._q[:, 1].astype(int) == uc) & \
                    (self._q[:, 2].astype(int) == hua)))[0].tolist()
                assert len(q_indices) == 2
                
                # use on-policy MC with an epsilon-greedy policy
                q_rows = np.array([self._q[ind, :] for ind in q_indices]) 
                # break ties randomly when using argmax
                if q_rows[0, 4] == q_rows[1, 4]: ind_max_q = random.randint(0, 1)
                else: ind_max_q = np.argmax(q_rows[:, 4], axis=0)
                maximizing_a = int(q_rows[ind_max_q, 3])
                other_a = int(q_rows[1 if ind_max_q == 0 else 0, 3])
                
                # columns for self._pi: {0=sum, 1=upcard, 2=usable ace, 3=action}, 4=probability
                pi_index = np.where(
                    (self._pi[:, 0].astype(int) == cs) & \
                    (self._pi[:, 1].astype(int) == uc) & \
                    (self._pi[:, 2].astype(int) == hua) & \
                    (self._pi[:, 3].astype(int) == maximizing_a))[0][0]
                self._pi[pi_index, 4] = 1 - EPSILON + EPSILON/2
                pi_index = np.where(
                    (self._pi[:, 0].astype(int) == cs) & \
                    (self._pi[:, 1].astype(int) == uc) & \
                    (self._pi[:, 2].astype(int) == hua) & \
                    (self._pi[:, 3].astype(int) == other_a))[0][0]
                self._pi[pi_index, 4] = EPSILON/2
        
        return self._pi, self._q

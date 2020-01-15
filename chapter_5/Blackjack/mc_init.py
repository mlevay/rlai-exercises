from datetime import timedelta
import numpy as np
import os
import random
import time
from typing import Dict

from .common import get_all_states, get_all_states_and_actions, pickle, unpickle
from .constants import DIR_ABS_PATH, DIR_REL_PATH_CTRL, DIR_REL_PATH_INIT, DIR_REL_PATH_PRED
from .constants import EPSILON
from .constants import PICKLE_FILE_NAME_EPISODES, PICKLE_FILE_NAME_STATS
from .constants import PLAYER_STICKS_AT, VERBOSE
from .game import Game
from .playback import Playback
from .stats import Stats, MCControlESStats, MCControlOnPolicyStats, MCPredictionStats


class MonteCarloInit():
    """
    (1) Initializes policy tables (random epsilon-soft policy or HIT20 policy);
    (2) Computes episodes (= simulated Blackjack games) for use with Monte Carlo.
    """
    def __init__(self, stats: Stats):
        assert isinstance(stats, MCControlESStats) == True or \
            isinstance(stats,  MCControlOnPolicyStats) == True or \
            isinstance(stats, MCPredictionStats) == True
        
        self.exploring_starts = isinstance(stats, MCControlESStats)
        self.stats = stats
        self.file_name_episodes, self.file_name_stats = self._get_file_paths()
        self.cols = self._init_cols(stats)
        
    def _init_cols(self, stats: Stats) -> Dict:
        if isinstance(stats, MCPredictionStats) == True:
            cols = {
                "cs": Stats.COL_CARD_SUM, 
                "uc": Stats.COL_UPCARD, 
                "hua": Stats.COL_HAS_USABLE_ACE, 
                "v(s)": MCPredictionStats.COL_V_OF_S, 
                "pi(s)": MCPredictionStats.COL_PI_OF_S, 
                "v_count": MCPredictionStats.COL_VISITS
            }
        elif isinstance(stats, MCControlESStats) == True:
            cols = {
                "cs": Stats.COL_CARD_SUM, 
                "uc": Stats.COL_UPCARD, 
                "hua": Stats.COL_HAS_USABLE_ACE, 
                "a": MCControlESStats.COL_A, 
                "q(s,a)": MCControlESStats.COL_Q_OF_S_A, 
                "pi(s)": MCControlESStats.COL_PI_OF_S, 
                "v_count": MCControlESStats.COL_VISITS, 
                "sv_count": MCControlESStats.COL_START_VISITS,
                "v(s)": MCControlESStats.COL_V_OF_S, 
                "prob(a|s)": MCControlESStats.COL_PROB
            }
        elif isinstance(stats, MCControlOnPolicyStats) == True:
            cols = {
                "cs": Stats.COL_CARD_SUM, 
                "uc": Stats.COL_UPCARD, 
                "hua": Stats.COL_HAS_USABLE_ACE, 
                "a": MCControlOnPolicyStats.COL_A, 
                "q(s,a)": MCControlOnPolicyStats.COL_Q_OF_S_A, 
                "pi(s,a)": MCControlOnPolicyStats.COL_PI_OF_S_A, 
                "v_count": MCControlOnPolicyStats.COL_VISITS
            }
        return cols
    
    def _get_file_paths(self) -> (str, str):
        eps_file_path, stats_file_path = "", "" 
        
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_INIT)
        eps_file_path = os.path.join(rel_path, PICKLE_FILE_NAME_EPISODES)
                
        if isinstance(self.stats, MCPredictionStats) == True:
            rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_PRED)
        elif isinstance(self.stats, MCControlESStats) == True or \
            isinstance(self.stats, MCControlOnPolicyStats) == True:
            rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_CTRL)
        stats_file_path = os.path.join(rel_path, PICKLE_FILE_NAME_STATS)
        
        return eps_file_path, stats_file_path   
    
    # def init_pi_of_s(self, player_sticks_at) -> np.ndarray:
    #     """
    #     Creates and returns a new policy function for the specified policy choice.
    #     """
    #     all_states = get_all_states()
        
    #     pi = np.zeros((len(all_states), 4), dtype=int)
    #     pi[:, :-1] = all_states
    #     pi[:, -1] = (pi[:, 0] < player_sticks_at).astype(int)

    #     return pi 
    
    # def init_pi_of_s_and_a(self, player_sticks_at) -> np.ndarray:
    #     """
    #     Creates and returns a new policy function for the specified policy choice.
    #     """
    #     all_states_and_actions = get_all_states_and_actions()
        
    #     pi = np.zeros((len(all_states_and_actions), 5), dtype=float)
    #     pi[:, :-1] = all_states_and_actions
    #     pi[(pi[:, 0].astype(int) < player_sticks_at) & (pi[:, 3] == 1), -1] = 1 - EPSILON
    #     pi[(pi[:, 0].astype(int) >= player_sticks_at) & (pi[:, 3] == 1), -1] = EPSILON
    #     pi[(pi[:, 0].astype(int) < player_sticks_at) & (pi[:, 3] == 0), -1] = EPSILON
    #     pi[(pi[:, 0].astype(int) >= player_sticks_at) & (pi[:, 3] == 0), -1] = 1 - EPSILON

    #     return pi

    def start_compute(self, commit_to_disk: bool=False):
        self.commit_to_disk = commit_to_disk
        
        self.playback = Playback(self.stats)
        self.game = Game(self.playback, exploring_starts=self.exploring_starts)
        
    def _play_one_game(self, pi: np.ndarray) -> Playback.Episode:
        outcome, episode = self.game.play(pi)

        if VERBOSE == True:
            print("Game outcome: {}".format(str(outcome).split(".")[-1]))
            print(episode.actors_k)
            print(episode.states_k_sum)
            print(episode.states_k_upcard_value)
            print(episode.states_k_has_usable_ace)
            print(episode.actions_k)
            print(episode.rewards_k_plus_1)
            print()
            
        return episode
    
    def compute_episode(self) -> np.ndarray:
        """
        Computes and returns a single episode (=Blackjack game) with the given policy.
        """
        episode = self._play_one_game(pi)
        
        if VERBOSE == True:
            print("Episode:")
            print(episode.actors_k)
            print(episode.states_k_sum)
            print(episode.states_k_upcard_value)
            print(episode.states_k_has_usable_ace)
            print(episode.actions_k)
            print(episode.rewards_k_plus_1)
            print()
            
        return episode
    
    def end_compute(self):
        if self.exploring_starts == True:
            print("Stats:")
            for sa_c in self.stats._stats:
                print("state=[{}, {}, {}], action={}, count={}".format(
                    sa_c[self.cols["cs"]], sa_c[self.cols["uc"]], sa_c[self.cols["hua"]], \
                    sa_c[self.cols["a"]], sa_c[self.cols["sv_count"]]))
                
        if self.commit_to_disk == True: self.save_episodes(self.playback.episodes)
    
    def load_episodes(self) -> np.ndarray:
        """
        Loads and returns the episodes array from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_episodes)
    
    def save_episodes(self, episodes):
        """
        Saves the episodes array to disk.
        """
        pickle(self.file_name_episodes, episodes)
    
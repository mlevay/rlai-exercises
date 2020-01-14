from datetime import timedelta
import numpy as np
import os
import random
import time

from .common import get_all_states, get_all_states_and_actions, pickle, unpickle
from .constants import DIR_ABS_PATH, DIR_REL_PATH_CTRL, DIR_REL_PATH_INIT
from .constants import EPSILON
from .constants import PICKLE_FILE_NAME_INIT_EPISODES, PICKLE_FILE_NAME_INIT_PI
from .constants import PLAYER_STICKS_AT, VERBOSE
from .game import Game
from .playback import Playback
from .stats import Stats, MCControlESStats


class MonteCarloInit():
    """
    (1) Initializes policy tables (random epsilon-soft policy or HIT20 policy);
    (2) Computes episodes (= simulated Blackjack games) for use with Monte Carlo.
    """
    def __init__(self, stats: Stats, exploring_starts: bool=False):
        self.exploring_starts = exploring_starts
        self.file_name_episodes, self.file_name_pi = self._get_file_paths()
        self.stats = stats
    
    def _get_file_paths(self) -> (str, str):
        eps_file_path, pi_file_path = "", "" 
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_CTRL) \
            if self.exploring_starts == True else os.path.join(DIR_ABS_PATH, DIR_REL_PATH_INIT)            
        eps_file_path = os.path.join(rel_path, PICKLE_FILE_NAME_INIT_EPISODES)
        
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_INIT)
        pi_file_path = os.path.join(rel_path, PICKLE_FILE_NAME_INIT_PI)
        
        return eps_file_path, pi_file_path   
    
    def init_pi_of_s(self, player_sticks_at) -> np.ndarray:
        """
        Creates and returns a new policy function for the specified policy choice.
        """
        all_states = get_all_states()
        
        pi = np.zeros((len(all_states), 4), dtype=int)
        pi[:, :-1] = all_states
        pi[:, -1] = (pi[:, 0] < player_sticks_at).astype(int)

        return pi 
    
    def init_pi_of_s_and_a(self, player_sticks_at) -> np.ndarray:
        """
        Creates and returns a new policy function for the specified policy choice.
        """
        all_states_and_actions = get_all_states_and_actions()
        
        pi = np.zeros((len(all_states_and_actions), 5), dtype=float)
        pi[:, :-1] = all_states_and_actions
        pi[(pi[:, 0].astype(int) < player_sticks_at) & (pi[:, 3] == 1), -1] = 1 - EPSILON
        pi[(pi[:, 0].astype(int) >= player_sticks_at) & (pi[:, 3] == 1), -1] = EPSILON
        pi[(pi[:, 0].astype(int) < player_sticks_at) & (pi[:, 3] == 0), -1] = EPSILON
        pi[(pi[:, 0].astype(int) >= player_sticks_at) & (pi[:, 3] == 0), -1] = 1 - EPSILON

        return pi

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
    
    def compute_episode(self, pi: np.ndarray) -> np.ndarray:
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
            for sa_c in self.stats:
                print("state=[{}, {}, {}], action={}, count={}".format(
                    sa_c[Stats.COL_CARD_SUM], sa_c[Stats.COL_UPCARD], sa_c[Stats.COL_HAS_USABLE_ACE], \
                    sa_c[MCControlESStats.COL_A], sa_c[MCControlESStats.COL_START_VISITS]))
                
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
    
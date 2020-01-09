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


class MonteCarloInit():
    """
    (1) Initializes a policy table (random epsilon-soft policy or HIT20 policy);
    (2) Computes a given number of episodes (= simulated Blackjack games) for use with Monte Carlo ES.
    """
    def __init__(self, soft_policy: bool = False, equal_probs: bool=False):
        self.equal_probs = equal_probs
        self.file_name_episodes, self.file_name_pi = self._get_file_paths()
        self._pi = self._init_pi(soft_policy)

    def _get_file_paths(self) -> (str, str):
        eps_file_path, pi_file_path = "", "" 
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_CTRL) \
            if self.equal_probs == True else os.path.join(DIR_ABS_PATH, DIR_REL_PATH_INIT)            
        eps_file_path = os.path.join(rel_path, PICKLE_FILE_NAME_INIT_EPISODES)
        
        rel_path = os.path.join(DIR_ABS_PATH, DIR_REL_PATH_INIT)
        pi_file_path = os.path.join(rel_path, PICKLE_FILE_NAME_INIT_PI)
        
        return eps_file_path, pi_file_path   
    
    def get_pi_of_s(self, player_sticks_at, commit_to_disk=False) -> np.ndarray:
        """
        Creates and returns a new policy function for the specified policy choice.
        """
        all_states = get_all_states()
        
        pi = np.zeros((len(all_states), 4), dtype=int)
        pi[:, :-1] = all_states
        pi[:, -1] = (pi[:, 0] < player_sticks_at).astype(int)

        if commit_to_disk == True: self._save_pi(pi)
        return pi 
    
    def get_pi_of_s_and_a(self, player_sticks_at, commit_to_disk=False) -> np.ndarray:
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

        if commit_to_disk == True: self._save_pi(pi)
        return pi
    
    def _init_pi(self, soft_policy: bool) -> np.ndarray:
        """
        Initializes, saves to disk and returns a new policy function.
        If soft_policy is True, this will create a random epsilon-soft policy;
        otherwise, it will create a deterministic HIT20 policy.
        """
        if soft_policy == False:
            pi = self.get_pi_of_s(PLAYER_STICKS_AT, commit_to_disk=True)
        else:
            pi = self.get_pi_of_s_and_a(PLAYER_STICKS_AT, commit_to_disk=True)
        return pi

    def load_pi(self) -> np.ndarray:
        """
        Loads and returns the policy function from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_pi)
    
    def _save_pi(self, pi):
        """
        Saves the policy function to disk.
        """
        pickle(self.file_name_pi, pi)
        
    def _play_one_game(self, playback: Playback, game: Game, cards: []=[]):
        outcome = game.play(cards=cards)

        if VERBOSE == True:
            print("Game outcome: {}".format(str(outcome).split(".")[-1]))
            if len(playback.episodes) > 0:
                print(playback.episodes[-1].actors_k)
                print(playback.episodes[-1].states_k_sum)
                print(playback.episodes[-1].states_k_upcard_value)
                print(playback.episodes[-1].states_k_has_usable_ace)
                print(playback.episodes[-1].actions_k)
                print(playback.episodes[-1].rewards_k_plus_1)
            print()
        
    def compute_episodes(self, cards: [], num_episodes: int, commit_to_disk=False, pi: np.ndarray = None) -> np.ndarray:
        assert not (self.equal_probs == True and len(cards) > 0)
        
        i, j = 0, 0
        if pi is None: pi = self._pi # if no override, use the HIT20 policy previously initialized
        
        start_time = time.time()
        
        playback = Playback()
        playback.start(pi)
        game = Game(playback, equal_probs=self.equal_probs)
        take_size, curr_take = num_episodes/10, 1
        while i < num_episodes:
            if len(cards) > j:
                self._play_one_game(playback, game, cards=cards[j])
                j += 1
            else:
                self._play_one_game(playback, game)
            if len(playback.episodes[-1].actors_k) > 0: 
                i += 1
                if i == curr_take*take_size: 
                    elapsed_time = time.time() - start_time
                    print("{} percent done. Elapsed time: {}".format(
                        10*curr_take, 
                        timedelta(seconds=elapsed_time)))
                    curr_take += 1
        playback.end()
        
        elapsed_time = time.time() - start_time
        print("Done. Elapsed time: {}".format(timedelta(seconds=elapsed_time)))

        if VERBOSE == True:
            print("All episodes:")
            for episode in playback.episodes:
                print("Episode:")
                print(episode.actors_k)
                print(episode.states_k_sum)
                print(episode.states_k_upcard_value)
                print(episode.states_k_has_usable_ace)
                print(episode.actions_k)
                print(episode.rewards_k_plus_1)
                
            print()
        if self.equal_probs == True:
            print("Stats:")
            for sa_c in game.player.stats:
                print("state=[{}, {}, {}], action={}, count={}".format(
                    sa_c[0], sa_c[1], sa_c[2], sa_c[3], sa_c[4]))
                
        if commit_to_disk == True: self._save_episodes(playback.episodes)
        return playback.episodes
    
    def load_episodes(self) -> np.ndarray:
        """
        Loads and returns the episodes array from disk, or None if it doesn't exist.
        """
        return unpickle(self.file_name_episodes)
    
    def _save_episodes(self, episodes):
        """
        Saves the episodes array to disk.
        """
        pickle(self.file_name_episodes, episodes)
    
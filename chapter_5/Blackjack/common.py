import enum
import numpy as np
import os
from pathlib import Path
import pickle as pckl

from .constants import MAX_CARD_SUM, MIN_CARD_SUM


def enum_to_string(item: enum.Enum) -> str:
    """
    Extracts the part of an enum item that is after the dot.
    """
    return str(item).split(".")[-1]           
        
def get_all_states() -> np.ndarray:
    """
    Returns a matrix representing all possible states, one row per state / one column per state variable.
    """
    all_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM+1))
    all_upcard_values = list(range(1, 11))
    all_ace_states = [False, True]
    all_states = np.array(np.meshgrid(all_sums, all_upcard_values, all_ace_states)).T.reshape(-1, 3)
    return all_states

def get_all_states_and_actions() -> np.ndarray:
    """
    Returns a matrix representing all possible states and actions, one row per state and action / one column per state variable or action.
    """
    all_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM+1))
    all_upcard_values = list(range(1, 11))
    all_ace_states = [False, True]
    
    all_actions = [0, 1]
    
    all_states_and_actions = np.array(np.meshgrid(all_sums, all_upcard_values, all_ace_states, all_actions)).T.reshape(-1, 4)
    return all_states_and_actions

def pickle(file_path, data):
    """
    Saves data in a pickle file. The directory will be created if necessary.
    """
    # double down on os.path.dirname as pickle files have no extensions
    # and are hence treated as folders
    folder_path = os.path.dirname(os.path.dirname(Path(file_path))) 
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    folder_path = os.path.dirname(Path(file_path))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    output_file = open(file_path, 'wb')
    pckl.dump(data, output_file)
    output_file.close()

def unpickle(file_path):
    """
    Loads and returns data from a pickle file, or None if the file doesn't exist.
    """
    data = None
    if os.path.exists(file_path):
        input_file = open(file_path, 'rb')
        data = pckl.load(input_file)
        input_file.close()
    return data
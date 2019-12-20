from datetime import datetime
import numpy as np
import os
import pandas as pd
import scipy.special

from .constants import PATH_SPRENRET_CSV

def print_status(text):
    print(text, datetime.now().strftime("%H:%M:%S"))
    
def get_state_name(state_a, state_b):
    """Obtain the unique state name in the format <num cars loc A>_<num cars loc B>"""
    return state_a.zfill(2) + "_" + state_b.zfill(2)# helper function to calculate the probability of rentals & returns at a location

def get_state_components(state_name):
    """Obtain the individual components (# cars at location A and B) for a state name"""
    return list(map(int, (state_name.split('_'))))

def commit_to_csv(df, file_name, dir_path=None):
    """Commit a dataframe to CSV file"""
    if dir_path == None: dir_path = PATH_SPRENRET_CSV
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    abs_file_name = os.path.join(dir_path, file_name)
    df.to_csv(path_or_buf=abs_file_name, sep='\t', encoding='utf-8', index=False)
    
def load_from_csv(file_name, dir_path=None):
    """Load a dataframe from CSV file"""
    if dir_path == None: dir_path = PATH_SPRENRET_CSV
    abs_file_name = os.path.join(dir_path, file_name)
    return pd.read_csv(filepath_or_buffer=abs_file_name, sep='\t', encoding='utf-8')

# module testing code
if __name__ == '__main__':
    pass

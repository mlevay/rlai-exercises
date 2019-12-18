from datetime import datetime
import numpy as np
import os
import pandas as pd
import scipy.special

from constants import *

def print_status(text):
    print(text, datetime.now().strftime("%H:%M:%S"))
    
def get_state_name(state_a, state_b):
    """Obtain the unique state name in the format <num cars loc A>_<num cars loc B>"""
    return state_a.zfill(2) + "_" + state_b.zfill(2)# helper function to calculate the probability of rentals & returns at a location

def commit_to_csv(df, file_name):
    """Commit a dataframe to CSV file"""
    if not os.path.exists(PATH_SPRENRET_CSV):
        os.makedirs(PATH_SPRENRET_CSV)

    abs_file_name = os.path.join(PATH_SPRENRET_CSV, file_name)
    df.to_csv(path_or_buf=abs_file_name, sep='\t', encoding='utf-8', index=False)
    
def load_from_csv(file_name):
    """Load a dataframe from CSV file"""
    abs_file_name = os.path.join(PATH_SPRENRET_CSV, file_name)
    return pd.read_csv(filepath_or_buffer=abs_file_name, sep='\t', encoding='utf-8')

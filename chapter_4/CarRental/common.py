from datetime import datetime
import enum
import numpy as np
import os
import pandas as pd
import scipy.special

from .constants import FILE_SASP_PREFIX, FILE_SPRENRET_PREFIX, FILE_PI_PREFIX, FILE_V_PREFIX
from .constants import FOLDER_ORIG_PROB_NAME, FOLDER_FULL_PROB_NAME
from .constants import PATH_SPRENRET_CSV


class FileType(enum.Enum):
    SASP = 1
    Sp_Ren_Ret = 2
    Pi = 3
    V = 4
    
def _get_filename_noext(file_type: FileType):
    file_name = ""
    if file_type == FileType.SASP: file_name = FILE_SASP_PREFIX
    elif file_type == FileType.Sp_Ren_Ret: file_name = FILE_SPRENRET_PREFIX
    elif file_type == FileType.Pi: file_name = FILE_PI_PREFIX
    elif file_type == FileType.V: file_name = FILE_V_PREFIX
    
    return file_name
    
def get_filename(file_type: FileType):
    file_name = _get_filename_noext(file_type)
    return file_name + ".csv"
    
def get_filename_with_postfix(file_type: FileType, seq_nr):
    file_name = _get_filename_noext(file_type)
    return file_name + str(seq_nr).zfill(2) + ".csv"

def print_status(text):
    print(text, datetime.now().strftime("%H:%M:%S"))
    
def get_state_name(state_a, state_b):
    """Obtain the unique state name in the format <num cars loc A>_<num cars loc B>"""
    return state_a.zfill(2) + "_" + state_b.zfill(2)# helper function to calculate the probability of rentals & returns at a location

def get_state_components(state_name):
    """Obtain the individual components (# cars at location A and B) for a state name"""
    comps = list(map(int, (state_name.split('_'))))
    return comps[0], comps[1]
    #return list(map(int, (state_name.split('_'))))

def commit_to_csv(df, file_type: FileType, is_orig_problem, seq_nr = 0, dir_path=None):
    """Commit a dataframe to CSV file"""
    suffix_file = file_type in [FileType.SASP, FileType.Sp_Ren_Ret]
    file_name = get_filename(file_type) if suffix_file else get_filename_with_postfix(file_type, seq_nr)
    
    if dir_path == None: dir_path = PATH_SPRENRET_CSV
    dir_path = os.path.join(dir_path, FOLDER_ORIG_PROB_NAME) if is_orig_problem == True else os.path.join(dir_path, FOLDER_FULL_PROB_NAME)
    if not os.path.exists(dir_path):
        print("Folder " + dir_path + " could not be found on disk, will create it.")
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path) == True:
        raise FileExistsError("A file with the directory name you specified already exists. Please resolve the issue and run the code again.")

    abs_file_name = os.path.join(dir_path, file_name)
    df.to_csv(path_or_buf=abs_file_name, sep='\t', encoding='utf-8', index=False)
    
def load_from_csv(file_type: FileType, is_orig_problem, seq_nr = 0, dir_path=None):
    """Load a dataframe from CSV file"""
    suffix_file = file_type in [FileType.SASP, FileType.Sp_Ren_Ret]
    file_name = get_filename(file_type) if suffix_file else get_filename_with_postfix(file_type, seq_nr)
        
    if dir_path == None: dir_path = PATH_SPRENRET_CSV
    dir_path = os.path.join(dir_path, FOLDER_ORIG_PROB_NAME) if is_orig_problem == True else os.path.join(dir_path, FOLDER_FULL_PROB_NAME)
    if os.path.exists(dir_path) == False: 
        print("Folder " + dir_path + " could not be found on disk.")
        return pd.DataFrame() # this will signal that the data we are looking for doesn't exist
    elif os.path.isfile(dir_path) == True:
        raise FileNotFoundError("A file with the directory name you specified already exists. Please resolve the issue and run the code again.")
    
    abs_file_name = os.path.join(dir_path, file_name)
    if os.path.exists(abs_file_name) == False:
        print("File " + abs_file_name + " could not be found on disk, will compute and create it instead.")
        return pd.DataFrame() # this will signal that the data we are looking for doesn't exist
    elif os.path.isdir(abs_file_name) == True:
        raise FileNotFoundError("A directory with the file name you specified already exists. Please resolve the issue and run the code again.")
    
    return pd.read_csv(filepath_or_buffer=abs_file_name, sep='\t', encoding='utf-8')


# module testing code
if __name__ == '__main__':
    pass

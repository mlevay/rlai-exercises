import numpy as np
import pandas as pd
from CarRental import common, compute, constants, postprocess, preprocess

def run():
    # the following will grab the values either from overrules or from the module itself
    is_orig_problem = constants.ORIGINAL_PROBLEM
    disk_allowed = constants.USE_DISK_FOR_CSV_DATA
    use_csv_data = constants.GET_DATA_FROM_CSV
    use_csv_model = constants.GET_MODEL_FROM_CSV
    dir_path = constants.PATH_SPRENRET_CSV
    pi_seq_nr = constants.PI_SEQ_NR
    v_seq_nr = constants.V_SEQ_NR
    
    print(is_orig_problem)

    dfSASP, dfSp_Ren_Ret, dfPi, dfV = None, None, None, None
    # Pre-process data, or get it from CSV files
    if disk_allowed == True:
        if use_csv_data == True:
            # get cached pre-processed data from disk
            dfSASP = common.load_from_csv("dfSASP.csv", dir_path=dir_path)
            dfSp_Ren_Ret = common.load_from_csv("dfSp_Ren_Ret.csv", dir_path=dir_path)
        else:
            # pre-process data and commit to to disk
            dfSASP = preprocess.prep_dfSASP(is_orig_problem=is_orig_problem)
            dfSp_Ren_Ret = preprocess.prep_dfSpRenRet(is_orig_problem=is_orig_problem)
            common.commit_to_csv(dfSASP, "dfSASP.csv", dir_path=dir_path)
            common.commit_to_csv(dfSp_Ren_Ret, "dfSp_Ren_Ret.csv", dir_path=dir_path)
    else:
        dfSASP = preprocess.prep_dfSASP(is_orig_problem=is_orig_problem)
        dfSp_Ren_Ret = preprocess.prep_dfSpRenRet(is_orig_problem=is_orig_problem)
        
    # Compute policy and value function
    if disk_allowed == True and use_csv_model == True and pi_seq_nr > -1 and v_seq_nr > -1:
        dfPi = common.load_from_csv("dfPi" + str(pi_seq_nr).zfill(2) + ".csv", dir_path=dir_path)
        dfV = common.load_from_csv("pfV" + str(v_seq_nr).zfill(2) + ".csv", dir_path=dir_path)
    else:
        dfPi, dfV = compute.policy_iteration(
            dfSASP, dfSp_Ren_Ret, pi_seq_nr=pi_seq_nr, v_seq_nr=v_seq_nr, 
            disk_allowed=disk_allowed, dir_path=dir_path)
    postprocess.transform_data(dfPi, dfV)
    

# module testing code
if __name__ == '__main__':
    run()
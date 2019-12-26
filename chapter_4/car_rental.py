import numpy as np
import os
import pandas as pd
from CarRental import common, compute, constants, plot, postprocess, preprocess

def run():
    # the following will grab the values either from overrules or from the module itself
    is_orig_problem = constants.ORIGINAL_PROBLEM
    disk_allowed = constants.USE_DISK_FOR_CSV_DATA
    use_csv_data = constants.GET_DATA_FROM_CSV
    use_csv_model = constants.GET_MODEL_FROM_CSV
    dir_path = constants.PATH_SPRENRET_CSV
    pi_seq_nr = constants.PI_SEQ_NR
    v_seq_nr = constants.V_SEQ_NR
    
    # initialize the four dataframes as empty dataframes, and use
    # if dataframe.empty() to check if data could be loaded from file
    dfSASP, dfSp_Ren_Ret, dfPi, dfV = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Pre-process from scratch, or get pre-processed data from CSV files
    create_dfSASP, create_dfSp_Ren_Ret = False, False
    if disk_allowed == True and use_csv_data == True:
        # get cached pre-processed data from disk
        dfSASP = common.load_from_csv(common.FileType.SASP, dir_path=dir_path)
        dfSp_Ren_Ret = common.load_from_csv(common.FileType.Sp_Ren_Ret, dir_path=dir_path)
        # if any of the files couldn't be found or accessed 
        if dfSASP.empty: create_dfSASP = True
        if dfSp_Ren_Ret.empty: create_dfSp_Ren_Ret = True
    elif disk_allowed == True and use_csv_data == False:
        create_dfSASP, create_dfSp_Ren_Ret = True, True
        
    if create_dfSASP == True:
        dfSASP = preprocess.prep_dfSASP(is_orig_problem=is_orig_problem)
        if disk_allowed == True: 
            common.commit_to_csv(dfSASP, common.FileType.SASP, dir_path=dir_path)
    if create_dfSp_Ren_Ret == True:
        dfSp_Ren_Ret = preprocess.prep_dfSpRenRet(is_orig_problem=is_orig_problem)
        if disk_allowed == True:
            common.commit_to_csv(dfSp_Ren_Ret, common.FileType.Sp_Ren_Ret, dir_path=dir_path)   
        
    # Compute policy and value function (Policy Iteration)
    if disk_allowed == True and use_csv_model == True:
        # get cached model from disk 
        if pi_seq_nr > -1:
            dfPi = common.load_from_csv(common.FileType.Pi, seq_nr=pi_seq_nr, dir_path=dir_path)
            if dfPi.empty: pi_seq_nr = -1 # the file couldn't be found or accessed 
        if v_seq_nr > -1:
            dfV = common.load_from_csv(common.FileType.V, seq_nr=v_seq_nr, dir_path=dir_path)
            if dfV.empty: v_seq_nr = -1 # the file couldn't be found or accessed 
    elif disk_allowed == True and use_csv_model == False:
        pi_seq_nr, v_seq_nr = -1, -1
    
    if pi_seq_nr == -1 or v_seq_nr == -1:
        dfPi, dfV = compute.policy_iteration(
            dfSASP, dfSp_Ren_Ret, pi_seq_nr=pi_seq_nr, v_seq_nr=v_seq_nr, 
            disk_allowed=disk_allowed, dir_path=dir_path)
        
    dfV_pivoted, dfPi_s_pivoted = postprocess.transform_data(dfPi, dfV)
    plot.plot_V(dfV_pivoted)
    plot.plot_Pi(dfPi_s_pivoted)
    

# module testing code
if __name__ == '__main__':
    run()
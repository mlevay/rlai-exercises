from datetime import datetime
import numpy as np
import pandas as pd

from common import *
from constants import *

def init_policy_iteration(pi_seq_nr=-1, v_seq_nr=-1):
    # Load data from CSV
    dfSASP = load_from_csv("dfSASP.csv")
    dfSp_Ren_Ret = load_from_csv("dfSp_Ren_Ret.csv")
    # this is just to add a new column whenever code changes 
    if not(DFCOL_SPRENRET_FEES in dfSp_Ren_Ret.columns):
        dfSp_Ren_Ret[DFCOL_SPRENRET_FEES] = [0]*dfSp_Ren_Ret.shape[0]
    
    # if we need to load one or both of the two dataframes (dfPi, dfV)
    if pi_seq_nr > -1:
        dfPi = load_from_csv("dfPi" + str(pi_seq_nr).zfill(2) + ".csv")
    if v_seq_nr > -1:
        dfV = load_from_csv("dfV" + str(v_seq_nr).zfill(2) + ".csv")
    
    # if we need to do calculation for any of the two dataframes (dfPi, dfV)
    if (pi_seq_nr == -1 or v_seq_nr == -1):
        # Calculate the row count for the dfS_A_Sp dataframe
        num_states = (MAX_NUMBER_OF_CARS_LOC_1 - MIN_NUMBER_OF_CARS_LOC_1 + 1) * (MAX_NUMBER_OF_CARS_LOC_2 - MIN_NUMBER_OF_CARS_LOC_2 + 1)

        # Calculate all states
        all_sub_states_a = list(range(MIN_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_1 + 1))
        all_sub_states_b = list(range(MIN_NUMBER_OF_CARS_LOC_2, MAX_NUMBER_OF_CARS_LOC_2 + 1))
        all_states = np.array(np.meshgrid(all_sub_states_a, all_sub_states_b)).T.reshape(-1,2)

        state_names = np.array(["xx_xx"] * num_states)
        for i in range(num_states):
            state_name = get_state_name(str(all_states[i,0].item()), str(all_states[i,1].item()))
            state_names[i] = state_name

        all_states = np.hstack((all_states, np.atleast_2d(state_names).T))
        
        if pi_seq_nr == -1:
            # Create dataframe dfPi - init with default action = 5 (= 0 transfers)
            dfPi = dfSASP[[DFCOL_SASP_SORIG, DFCOL_SASP_ACTION]].copy()
            dfPi.columns = [DFCOL_PI_STATE, DFCOL_PI_ACTION]
            dfPi[DFCOL_PI_PROB] = [0.]*dfPi.shape[0]
            
            # create a dict of format {state:df_actions_and_probs}
            groups = dict(list(dfPi.groupby(DFCOL_PI_STATE)))
            for s in groups:
                dfForState = groups[s]
                num_actions = float(dfForState.shape[0])
                prob_pref_action = 1. - EPSILON + (EPSILON/num_actions)
                prob_other_action = EPSILON/num_actions
                dfPi.loc[(dfPi[DFCOL_PI_STATE] == s) & (dfPi[DFCOL_PI_ACTION] == DEFAULT_ACTION), DFCOL_PI_PROB] = prob_pref_action
                dfPi.loc[(dfPi[DFCOL_PI_STATE] == s) & (dfPi[DFCOL_PI_ACTION] != DEFAULT_ACTION), DFCOL_PI_PROB] = prob_other_action
            
            dfPi.set_index([DFCOL_PI_STATE, DFCOL_PI_ACTION], inplace=True)    
            dfPi.sort_index(axis=0, inplace=True)
            dfPi.reset_index(inplace=True)
            print("(re)set index for dfPi", datetime.now().strftime("%H:%M:%S"))

        if v_seq_nr == -1:
            # Create dataframe dfV - init with default value = 0
            mindex = pd.MultiIndex.from_product([all_states[:,-1]], names=[DFCOL_V_STATE])
            dfV = pd.DataFrame(
                {
                    DFCOL_V_VALUE: [DEFAULT_VALUE]*num_states
                },
                index = mindex)
            print("initialized dataframe dfV", datetime.now().strftime("%H:%M:%S"))

            dfV.reset_index(inplace=True)
            #dfPi.set_index(DFCOL_V_STATE)
            print("(re)set index for dfV", datetime.now().strftime("%H:%M:%S"))
    
    return dfSASP, dfSp_Ren_Ret, dfV, dfPi, max(pi_seq_nr, v_seq_nr) + 1
    
def policy_evaluation(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr):
    while True:
        delta = 0.
        v, new_v = 0., 0.
        
        for index, row in dfV.iterrows():
            # the current state, interpreted as original state, and its current value
            orig_state, v = row[DFCOL_V_STATE], row[DFCOL_V_VALUE]
                
            # init the new value at 0. to keep adding partial values to it later
            new_v = 0.
            
            # legitimate actions for the state vary dependent on mode
            # we'll iterate through all legitimate actions, ignoring current greedy policy
            legit_actions = dfSASP.loc[dfSASP[DFCOL_SASP_SORIG] == orig_state, DFCOL_SASP_ACTION].tolist()
                
            for action in legit_actions:
                # the pseudo-state the original state and current action lead to, and the
                # penalty incurred by that action
                pseudo_state, action_penalty = dfSASP.loc[
                    (dfSASP[DFCOL_SASP_SORIG] == orig_state) & (dfSASP[DFCOL_SASP_ACTION] == action), 
                    [DFCOL_SASP_SPSEUDO, DFCOL_SASP_FEES]].values[0]
                
                # dataframe dfProbs holds the following columns from dfSp_Ren_Ret that correspond to the pseudo-state:
                # (1) all valid rental&return count combinations across locations
                # DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B, DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B, 
                # (2) their corresponding next state
                # DFCOL_SPRENRET_SNEXT
                # (3) their respective Poisson probabilities
                # DFCOL_SPRENRET_PROBSRSA, 
                # (4) their respective reward
                # DFCOL_SPRENRET_REWARD
                dfProbs = dfSp_Ren_Ret.loc[
                    dfSp_Ren_Ret[DFCOL_SPRENRET_SPSEUDO] == pseudo_state, 
                    [DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B, 
                    DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B, 
                    DFCOL_SPRENRET_SNEXT,
                    DFCOL_SPRENRET_PROBSRSA, 
                    DFCOL_SPRENRET_REWARD,
                    DFCOL_SPRENRET_FEES]]
                
                # loop through dfProbs and add to the state's value the probability-weighted average of 
                # (reward plus discounted next-state value) over next states and their rewards
                dfJoined = pd.merge(dfProbs, dfV, how="left", left_on=DFCOL_SPRENRET_SNEXT, right_on=DFCOL_V_STATE)
                prob_pi = dfPi.loc[(dfPi[DFCOL_PI_STATE] == orig_state) & (dfPi[DFCOL_PI_ACTION] == action), DFCOL_PI_PROB].values[0]
                new_v += ((
                    dfJoined[DFCOL_SPRENRET_PROBSRSA] * (
                        dfJoined[DFCOL_SPRENRET_REWARD] - action_penalty - dfJoined[DFCOL_SPRENRET_FEES] + GAMMA*dfJoined[DFCOL_V_VALUE]
                    )
                ).sum() * prob_pi)
            
            # update dfV with the new value for this state
            dfV.loc[index, DFCOL_V_VALUE] = new_v
                
            # keep record of the greatest yet delta between an old and a new state value within this iteration 
            delta = max(delta, abs(v - new_v))
        
        if(delta - THETA < 0.): 
            # computed values are good enough
            commit_to_csv(dfV, "dfV" + str(seq_nr).zfill(2) + ".csv")
            seq_nr = seq_nr + 1
            print_status("values deemed good enough")
            break 
        else:
            # computed values are not yet good enough
            print_status("values deemed not good enough, going for another value loop")
            print(dfV, delta)
            
    # there's a new value function => improve the policy next
    dfPi, dfV = policy_improvement(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)
    return dfPi, dfV
    
def policy_improvement(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr):
    delta = 0.
    v, new_v = 0., 0.
    policy_stable = True
    max_v = -100000.
    
    # create a dict of format {state:df_actions_and_probs}
    groups = dict(list(dfPi.groupby(DFCOL_PI_STATE)))
    for orig_state in groups:
        dfState = groups[orig_state]
    #for row in dfPi.iterrows():
        # the current state, interpreted as original state
        #orig_state = row[DFCOL_PI_STATE]
        
        max_v = -100000.
        new_a = None
        delta = 0.
        
        # the so far known maximizing action for this state
        #dfState = dfPi.loc[dfPi[DFCOL_PI_STATE] == orig_state, [DFCOL_PI_ACTION, DFCOL_PI_PROB]]
        a = dfState.loc[dfState[DFCOL_PI_PROB] == np.amax(dfState[DFCOL_PI_PROB]), DFCOL_PI_ACTION].values[0]
        v = dfV.loc[dfV[DFCOL_V_STATE] == orig_state, DFCOL_V_VALUE].values[0]
        
        # legitimate actions for the state vary dependent on mode
        # we'll iterate through all legitimate actions, ignoring current greedy policy
        legit_actions = dfState[DFCOL_PI_ACTION].tolist()
            
        for action in legit_actions:
            # the pseudo-state the original state and current action lead to, and the
            # penalty incurred by that action
            pseudo_state, action_penalty = dfSASP.loc[
                (dfSASP[DFCOL_SASP_SORIG] == orig_state) & (dfSASP[DFCOL_SASP_ACTION] == action), 
                [DFCOL_SASP_SPSEUDO, DFCOL_SASP_FEES]].values[0]
            
            # dataframe dfProbs holds the following columns from dfSp_Ren_Ret that correspond to the pseudo-state:
            # (1) all valid rental&return count combinations across locations
            # DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B, DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B, 
            # (2) their corresponding next state
            # DFCOL_SPRENRET_SNEXT
            # (3) their respective Poisson probabilities
            # DFCOL_SPRENRET_PROBSRSA, 
            # (4) their respective reward
            # DFCOL_SPRENRET_REWARD
            dfProbs = dfSp_Ren_Ret.loc[
                dfSp_Ren_Ret[DFCOL_SPRENRET_SPSEUDO] == pseudo_state, 
                [DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B, 
                 DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B, 
                 DFCOL_SPRENRET_SNEXT,
                 DFCOL_SPRENRET_PROBSRSA, 
                 DFCOL_SPRENRET_REWARD,
                 DFCOL_SPRENRET_FEES]]

            # loop through dfProbs and look for the action that would maximize the state's value
            new_v = 0.
            
            dfJoined = pd.merge(dfProbs, dfV, how="left", left_on=DFCOL_SPRENRET_SNEXT, right_on=DFCOL_V_STATE)
            new_v = (
                dfJoined[DFCOL_SPRENRET_PROBSRSA] * (
                    dfJoined[DFCOL_SPRENRET_REWARD] - action_penalty - dfJoined[DFCOL_SPRENRET_FEES] + GAMMA*dfJoined[DFCOL_V_VALUE]
                )
            ).sum()

            # if the computed state value is larger than seen so far, then the current action is a maximizer
            if max_v - new_v < 0.:
                max_v = new_v
                new_a = action
                
                # keep record of the greatest yet delta between an old and a new state value within this state loop 
                delta = max(delta, abs(v - new_v))
                
        # update the probabilities for the current state and all its possible actions in dfPi
        num_actions = float(dfPi[dfPi[DFCOL_PI_STATE] == orig_state].shape[0])
        prob_pref_action = 1. - EPSILON + (EPSILON/num_actions)
        prob_other_action = EPSILON/num_actions
        dfPi.loc[(dfPi[DFCOL_PI_STATE] == orig_state) & (dfPi[DFCOL_PI_ACTION] == new_a), DFCOL_PI_PROB] = prob_pref_action
        dfPi.loc[(dfPi[DFCOL_PI_STATE] == orig_state) & (dfPi[DFCOL_PI_ACTION] != new_a), DFCOL_PI_PROB] = prob_other_action
        
        if a != new_a and delta - THETA > 0.: 
            policy_stable = False
    
    #commit_to_csv(dfPi, "dfPi" + str(seq_nr).zfill(2) + ".csv")
    if policy_stable == True:
        # the policy is considered stable enough, so last computed values remains valid
        print_status("policy considered stable enough")
        print(dfPi)
        pass
    else:
        # a better policy was found, so policy evaluation must update the last computed values
        commit_to_csv(dfPi, "dfPi" + str(seq_nr).zfill(2) + ".csv")
        seq_nr = seq_nr + 1
        
        print_status("a better policy was found, going for another value loop")
        print(dfPi)
        dfPi, dfV = policy_evaluation(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)
        
    return dfPi, dfV

# module testing code
if __name__ == '__main__':
    v_seq_nr, pi_seq_nr = 2, 1
    
    assert (abs(pi_seq_nr - v_seq_nr) == 1) or (pi_seq_nr == -1 and v_seq_nr == -1)
    
    dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr = init_policy_iteration(pi_seq_nr=pi_seq_nr, v_seq_nr=v_seq_nr)
    if (pi_seq_nr > v_seq_nr) or (pi_seq_nr == -1 and v_seq_nr == -1):
        dfPi, dfV = policy_evaluation(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)
    elif pi_seq_nr < v_seq_nr:
        dfPi, dfV = policy_improvement(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)
    
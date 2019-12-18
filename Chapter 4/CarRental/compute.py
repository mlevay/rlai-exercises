from datetime import datetime
import numpy as np
import pandas as pd

from common import *
from constants import *

def init_policy_iteration(pi_seq_nr=-1, v_seq_nr=-1):
    # Load data from CSV
    dfSASP = load_from_csv("dfSASP.csv")
    dfSp_Ren_Ret = load_from_csv("dfSp_Ren_Ret.csv")
    
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
            mindex = pd.MultiIndex.from_product([all_states[:,-1]], names=[DFCOL_PI_STATE])
            dfPi = pd.DataFrame(
                {
                    DFCOL_PI_ACTION: [5]*num_states
                },
                index = mindex)
            print("initialized dataframe dfPi", datetime.now().strftime("%H:%M:%S"))

            dfPi.reset_index(inplace=True)
            #dfPi.set_index(DFCOL_PI_STATE)
            print("(re)set index for dfPi", datetime.now().strftime("%H:%M:%S"))

        if v_seq_nr == -1:
            # Create dataframe dfV - init with default value = 0
            mindex = pd.MultiIndex.from_product([all_states[:,-1]], names=[DFCOL_V_STATE])
            dfV = pd.DataFrame(
                {
                    DFCOL_V_VALUE: [0.]*num_states
                },
                index = mindex)
            print("initialized dataframe dfV", datetime.now().strftime("%H:%M:%S"))

            dfV.reset_index(inplace=True)
            #dfPi.set_index(DFCOL_V_STATE)
            print("(re)set index for dfV", datetime.now().strftime("%H:%M:%S"))
    
    return dfSASP, dfSp_Ren_Ret, dfV, dfPi
    
def policy_evaluation(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr):
    while True:
        delta, gamma, theta = 0., 0.9, 0.1
        v, new_v = 0., 0.
        
        for index, row in dfV.iterrows():
            # the current state, interpreted as original state, and its current value
            orig_state, v = row[DFCOL_V_STATE], row[DFCOL_V_VALUE]
                
            # init the new value at 0. to keep adding partial values to it later
            new_v = 0.
            
            # the action currently produced by the greedy policy for the original state
            action = dfPi.loc[dfPi[DFCOL_PI_STATE] == orig_state, DFCOL_PI_ACTION].values[0]

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
                 DFCOL_SPRENRET_REWARD]]
            
            # loop through dfProbs and add to the state's value the probability-weighted average of 
            # (reward plus discounted next-state value) over next states and their rewards
            dfJoined = pd.merge(dfProbs, dfV, how="left", left_on=DFCOL_SPRENRET_SNEXT, right_on=DFCOL_V_STATE)
            new_v = (
                dfJoined[DFCOL_SPRENRET_PROBSRSA] * (
                    dfJoined[DFCOL_SPRENRET_REWARD] - action_penalty + gamma*dfJoined[DFCOL_V_VALUE]
                )
            ).sum()
            
            # update dfV with the new value for this state
            dfV.loc[index, DFCOL_V_VALUE] = new_v
#                print("new value found for state " + orig_state + ": " + str(dfV.loc[index, DFCOL_V_VALUE]))
                
            # keep record of the greatest yet delta between an old and a new state value within this iteration 
            delta = max(delta, abs(v - new_v))
            #print("delta = " + str(delta))
        
        if(delta - theta < 0.): 
            # good enough
            commit_to_csv(dfV, "dfV" + str(seq_nr).zfill(2) + ".csv")
            seq_nr = seq_nr + 1
            print_status("values deemed good enough")
            break 
        else:
            # not yet good enough
            print_status("values deemed not good enough, going for another value loop")
            print(dfV)
            
    # there's a new value function => improve the policy next
    policy_improvement(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)
    
def policy_improvement(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr):
    gamma = 0.9
    v, new_v = 0., 0.
    policy_stable = True
    maximizing_action = None
    maximum_value = -100000.
    
    for index, row in dfPi.iterrows():
        # the current state, interpreted as original state
        orig_state = row[DFCOL_PI_STATE]
#        print("orig_state = " + orig_state)
        
        maximum_value = -100000.
        maximizing_action = None
        
        # the action currently produced by the greedy policy for the original state
        action = dfPi.loc[dfPi[DFCOL_PI_STATE] == orig_state, DFCOL_PI_ACTION].values[0]
#        print("action = " + str(action))

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
                 DFCOL_SPRENRET_REWARD]]

            # loop through dfProbs and look for the action that would maximize the state's value
            new_v = 0.
            
            dfJoined = pd.merge(dfProbs, dfV, how="left", left_on=DFCOL_SPRENRET_SNEXT, right_on=DFCOL_V_STATE)
            new_v = (
                dfJoined[DFCOL_SPRENRET_PROBSRSA] * (
                    dfJoined[DFCOL_SPRENRET_REWARD] - action_penalty + gamma*dfJoined[DFCOL_V_VALUE]
                )
            ).sum()

            # if the computed state value is larger than seen so far, then the current action is a maximizer
            if maximum_value - new_v < 0.:
                maximum_value = max(maximum_value, new_v)
                maximizing_action = action

        # record any new maximizing action for the state
        dfPi.loc[index, DFCOL_PI_ACTION] = maximizing_action
        new_v = maximum_value
            
        if action != maximizing_action and v < new_v: 
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
        policy_evaluation(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)

# module testing code
if __name__ == '__main__':
    #dfSASP = load_from_csv("dfSASP.csv")
    #print(dfSASP.head(20))

    #dfSp_Ren_Ret = load_from_csv("dfSp_Ren_Ret.csv")
    #print(dfSp_Ren_Ret.head(20))
    
    seq_nr = 0
    
    dfSASP, dfSp_Ren_Ret, dfV, dfPi = init_policy_iteration(pi_seq_nr=-1, v_seq_nr=-1)
    policy_evaluation(dfSASP, dfSp_Ren_Ret, dfV, dfPi, seq_nr)
    
from common import *
from constants import *
from probabilities import * 

def get_reward(rentals):
    """Calculate the reward from rentals across locations"""
    return rentals * REWARD_PER_RENTAL

def init_temp_df(data):
    """Initialize a temporary dataframe for SpRenRet computation"""
    columns=[
        DFCOL_SPRENRET_SPSEUDO,
        DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B,
        DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B,
        "p_ren_1", "p_ren_2", 
        "p_ret_1", "p_ret_2", 
        DFCOL_SPRENRET_PROBSRSA,
        DFCOL_SPRENRET_REWARD]
    
    dftemp = pd.DataFrame(data, columns=columns)
    
    dftemp = dftemp.astype({
        DFCOL_SPRENRET_SPSEUDO: str, 
        DFCOL_SPRENRET_RENTALS_A: int,
        DFCOL_SPRENRET_RENTALS_B: int,
        DFCOL_SPRENRET_RETURNS_A: int,
        DFCOL_SPRENRET_RETURNS_B: int,
        "p_ren_1": float, 
        "p_ren_2": float, 
        "p_ret_1": float, 
        "p_ret_2": float, 
        DFCOL_SPRENRET_PROBSRSA: float,
        DFCOL_SPRENRET_REWARD: int
    })
    
    return dftemp

num_states = 0
all_sub_states_a, all_sub_states_b = [], []
all_states = []
dict_states_a, dict_states_b = {}, {}

def prep_dfSASP():
    """
    Populate a dataframe with all valid combinations of state and action taken following that state. 
    Validity is bound by the minimum and maximum number of cars each location can host, as well as 
    the maximum cars that can be transferred across locations in a single go.
    Introduces the concept of a pseudo-state, which corresponds to the number of cars at loc A + B
    at 6am in the morning (= after transfer of cars across locations but prior to any new rentals/returns).
    
    Dataframe columns:
    s_k         The original state
    a_k         The action taken from s_k
    s_pseudo_k  The pseudo-state (see explanation above)
    fees_k      The fees incurred by the transfer corresponding to a_k
    """
    # Calculate the row count for the dfS_A_Sp dataframe
    num_states = (MAX_NUMBER_OF_CARS_LOC_1 - MIN_NUMBER_OF_CARS_LOC_1 + 1) * (MAX_NUMBER_OF_CARS_LOC_2 - MIN_NUMBER_OF_CARS_LOC_2 + 1)
    num_actions = MAX_NUMBER_OF_CARS_PER_TRANSFER*2 + 1 # all whole numbers in [-n, n]
    num_sasp = num_states * num_actions

    # Calculate all states
    all_sub_states_a = list(range(MIN_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_1 + 1))
    all_sub_states_b = list(range(MIN_NUMBER_OF_CARS_LOC_2, MAX_NUMBER_OF_CARS_LOC_2 + 1))
    all_states = np.array(np.meshgrid(all_sub_states_a, all_sub_states_b)).T.reshape(-1,2)

    state_names = np.array(["xx_xx"] * num_states)
    for i in range(num_states):
        state_name = get_state_name(str(all_states[i,0].item()), str(all_states[i,1].item()))
        state_names[i] = state_name

    all_states = np.hstack((all_states, np.atleast_2d(state_names).T))
    dict_states_a = dict(zip(all_states[:,-1], all_states[:,0]))
    dict_states_b = dict(zip(all_states[:,-1], all_states[:,1]))

    # Calculate all actions
    all_actions = np.array([list(range(-MAX_NUMBER_OF_CARS_PER_TRANSFER, MAX_NUMBER_OF_CARS_PER_TRANSFER+1))])

    action_names = np.array([0] * num_actions)
    for i in range(num_actions):
        action_name = i
        action_names[i] = action_name
        
    all_actions = np.hstack((all_actions.T, np.atleast_2d(action_names).T))
    dict_actions = dict(zip(all_actions[:,-1], all_actions[:,0]))

    mindex = pd.MultiIndex.from_product(
                    [all_states[:,-1], all_actions[:,-1]],
                    names=[DFCOL_SASP_SORIG, DFCOL_SASP_ACTION]
                )
    dfSASP = pd.DataFrame(
                {
                    DFCOL_SASP_IS_VALID: [True]*num_sasp
                },
                index = mindex)

    dfSASP.reset_index(inplace=True)

    # compute # cars at location A for the pseudo state: DFCOL_SARS_SPSEUDO_A
    dfSASP[DFCOL_SASP_SPSEUDO_A] = dfSASP[DFCOL_SASP_SORIG].map(dict_states_a).astype(int) - dfSASP[DFCOL_SASP_ACTION].map(dict_actions).astype(int)

    # compute # cars at location B for the pseudo state': DFCOL_SARS_SPSEUDO_B
    dfSASP[DFCOL_SASP_SPSEUDO_B] = dfSASP[DFCOL_SASP_SORIG].map(dict_states_b).astype(int) + dfSASP[DFCOL_SASP_ACTION].map(dict_actions).astype(int)

    # Re-cast the columns as specific types to allow string operations
    dfSASP = dfSASP.astype({DFCOL_SASP_SPSEUDO_A: str, 
                            DFCOL_SASP_SPSEUDO_B: str})

    # compute the pseudo state (as of 6am - following transfers but prior to new rentals/returns)
    dfSASP[DFCOL_SASP_SPSEUDO] = dfSASP.apply(
        lambda d: get_state_name(str(d[DFCOL_SASP_SPSEUDO_A]), str(d[DFCOL_SASP_SPSEUDO_B])),
        axis=1)

    # Re-cast the columns as specific types to allow arithmetic operations
    dfSASP = dfSASP.astype({DFCOL_SASP_SPSEUDO_A: int, 
                            DFCOL_SASP_SPSEUDO_B: int})

    dfSASP.loc[dfSASP[DFCOL_SASP_SPSEUDO_A] < MIN_NUMBER_OF_CARS_LOC_1, [DFCOL_SASP_IS_VALID]] = False
    dfSASP.loc[dfSASP[DFCOL_SASP_SPSEUDO_B] < MIN_NUMBER_OF_CARS_LOC_2, [DFCOL_SASP_IS_VALID]] = False
    dfSASP.loc[dfSASP[DFCOL_SASP_SPSEUDO_A] > MAX_NUMBER_OF_CARS_LOC_1, [DFCOL_SASP_IS_VALID]] = False
    dfSASP.loc[dfSASP[DFCOL_SASP_SPSEUDO_B] > MAX_NUMBER_OF_CARS_LOC_2, [DFCOL_SASP_IS_VALID]] = False

    dfSASP = dfSASP.loc[dfSASP[DFCOL_SASP_IS_VALID] == True]
    dfSASP = dfSASP.iloc[:,[0,1,5]]

    dfSASP = dfSASP.sort_values(DFCOL_SASP_SPSEUDO)

    # compute fees for a
    dfSASP[DFCOL_SASP_FEES] = abs(dfSASP[DFCOL_SASP_ACTION].map(dict_actions).astype(int)*UNIT_COST_OF_TRANSFER)
    
    return dfSASP

def prep_dfSpRenRet():
    """
    Populate a dataframe with all valid combinations of pseudo-state and rentals/returns. Validity is bound 
    by the minimum and maximum number of cars each location can host.
    Utilizes the concept of a pseudo-state, which corresponds to the number of cars at loc A + B
    at 6am in the morning (= after transfer of cars across locations but prior to any new rentals/returns).
    
    Dataframe columns:
    s_pseudo_k          The pseudo-state (see explanation above)
    rentals_k_a         Number of rentals at location A
    rentals_k_b         Number of rentals at location B
    returns_k_a         Number of returns at location A
    returns_k_b         Number of returns at location B
    p_of_srsa           The probability p(s', r | pseudo_s)
    r_k_reward          The total reward from rentals at both locations
    s_k_plus_1_a        The resulting number of cars at location A at the end of the day
    s_k_plus_1_b        The resulting number of cars at location B at the end of the day
    s_k_plus_1          The next state s'
    """
    # Calculate the row count for the dfSp_Ren_Ret dataframe
    num_states = (MAX_NUMBER_OF_CARS_LOC_1 - MIN_NUMBER_OF_CARS_LOC_1 + 1) * (MAX_NUMBER_OF_CARS_LOC_2 - MIN_NUMBER_OF_CARS_LOC_2 + 1)
    #num_ren_values = (MAX_NUMBER_OF_CARS_LOC_1 - MIN_NUMBER_OF_CARS_LOC_1 + 1) * (MAX_NUMBER_OF_CARS_LOC_2 - MIN_NUMBER_OF_CARS_LOC_2 + 1)
    #num_ret_values = (MAX_NUMBER_OF_CARS_LOC_1 - MIN_NUMBER_OF_CARS_LOC_1 + 1) * (MAX_NUMBER_OF_CARS_LOC_2 - MIN_NUMBER_OF_CARS_LOC_2 + 1)
    #num_psrr = num_states * num_ren_values * num_ret_values
    
    # Calculate all states
    all_sub_states_a = list(range(MIN_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_1 + 1))
    all_sub_states_b = list(range(MIN_NUMBER_OF_CARS_LOC_2, MAX_NUMBER_OF_CARS_LOC_2 + 1))
    all_states = np.array(np.meshgrid(all_sub_states_a, all_sub_states_b)).T.reshape(-1,2)

    state_names = np.array(["xx_xx"] * num_states)
    for i in range(num_states):
        state_name = get_state_name(str(all_states[i,0].item()), str(all_states[i,1].item()))
        state_names[i] = state_name

    all_states = np.hstack((all_states, np.atleast_2d(state_names).T))
    dict_states_a = dict(zip(all_states[:,-1], all_states[:,0]))
    dict_states_b = dict(zip(all_states[:,-1], all_states[:,1]))

    # Calculate all rental and return values
    all_renret_values_a = np.array(list(range(MIN_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_1 + 1)))
    all_renret_values_b = np.array(list(range(MIN_NUMBER_OF_CARS_LOC_2, MAX_NUMBER_OF_CARS_LOC_2 + 1)))
    print_status("pre-processing done")

    # Calculate all valid combinations of pseudo state, returns and rentals
    dfSp_Ren_Ret = pd.DataFrame(columns=[
        DFCOL_SPRENRET_SPSEUDO,
        DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B,
        DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B])

    ren_a, ren_b = [], []
    ret_a, ret_b = [], []

    for ps_a in range(MIN_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_1+1):
        for ps_b in range(MIN_NUMBER_OF_CARS_LOC_2, MAX_NUMBER_OF_CARS_LOC_2+1):
            ren_a = range(ps_a+1)
            ren_b = range(ps_b+1)
            ret_a = range(MAX_NUMBER_OF_CARS_LOC_1-ps_a+1)
            ret_b = range(MAX_NUMBER_OF_CARS_LOC_2-ps_b+1)
            
            cart_prod = np.array(np.meshgrid(
                [get_state_name(str(ps_a), str(ps_b))], 
                list(ren_a), list(ren_b), list(ret_a), list(ret_b)
            )).T.reshape(-1,5)
            rest = np.zeros(shape=(cart_prod.shape[0], 6), dtype='int')
            cart_prod = np.concatenate((cart_prod, rest), axis=1)
            
            # re-initialize the dftemp data frame
            dftemp = init_temp_df(cart_prod)
            
            # write the granular probabilities
            exp_numbers = [EXP_VALUE_RENTALS_LOC_1, EXP_VALUE_RENTALS_LOC_2, EXP_VALUE_RETURNS_LOC_1, EXP_VALUE_RETURNS_LOC_2]
            for i in range(4):
                probs = lookup_prob_vectorized(dftemp.iloc[:, 1+i], exp_numbers[i])
                dftemp.iloc[:, 5+i] = probs
            
#            # apply a softmax over each probability row
#            for i in range(4):
#                probs = softmax_vectorized(dftemp.iloc[:, 5+i])
#                dftemp.iloc[:, 5+i] = probs
                
            # compute and write the joint probability
            prob_rentals = np.multiply(dftemp.iloc[:, -6:-5], dftemp.iloc[:, -5:-4])
            prob_returns = np.multiply(dftemp.iloc[:, -4:-3], dftemp.iloc[:, -3:-2])
            dftemp.iloc[:, -2:-1] = np.multiply(prob_rentals.iloc[:,0], prob_returns.iloc[:,0])
            
            # retire the granular probability columns
            dftemp = dftemp.iloc[:, [0,1,2,3,4,9,10]]
            
            # compute the rewards for rentals
            dftemp.iloc[:, -1] = get_reward(dftemp.iloc[:, 1] + dftemp.iloc[:, 2])   
            # print(dftemp)
            
            dfSp_Ren_Ret = dfSp_Ren_Ret.append(pd.DataFrame(dftemp, columns=[
                DFCOL_SPRENRET_SPSEUDO,
                DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B,
                DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B,
                DFCOL_SPRENRET_PROBSRSA,
                DFCOL_SPRENRET_REWARD]))
            #print(dfSp_Ren_Ret) 
    print_status("dataframe creation done")
    
    dfSp_Ren_Ret.set_index([DFCOL_SPRENRET_SPSEUDO,
                DFCOL_SPRENRET_RENTALS_A, DFCOL_SPRENRET_RENTALS_B,
                DFCOL_SPRENRET_RETURNS_A, DFCOL_SPRENRET_RETURNS_B], inplace=True)
    dfSp_Ren_Ret.sort_index(inplace=True)
    dfSp_Ren_Ret.reset_index(inplace=True)
    print_status("dataframe multi-index set, sorted and removed")

    # compute the next state, s', from (s_pseudo, rentals_a, rentals_b, returns_a, returns_b)
    dfSp_Ren_Ret[DFCOL_SPRENRET_SNEXT_A] = \
        dfSp_Ren_Ret[DFCOL_SPRENRET_SPSEUDO].map(dict_states_a).astype(int) - \
        dfSp_Ren_Ret[DFCOL_SPRENRET_RENTALS_A] + dfSp_Ren_Ret[DFCOL_SPRENRET_RETURNS_A]
    dfSp_Ren_Ret[DFCOL_SPRENRET_SNEXT_B] = \
        dfSp_Ren_Ret[DFCOL_SPRENRET_SPSEUDO].map(dict_states_b).astype(int) - \
        dfSp_Ren_Ret[DFCOL_SPRENRET_RENTALS_B] + dfSp_Ren_Ret[DFCOL_SPRENRET_RETURNS_B]
    print_status("next state computed")

    # Compute the next state's name
    dfSp_Ren_Ret[DFCOL_SPRENRET_SNEXT] = dfSp_Ren_Ret.apply(
        lambda d: get_state_name(str(d[DFCOL_SPRENRET_SNEXT_A]), str(d[DFCOL_SPRENRET_SNEXT_B])),
        axis=1)
    print_status("next state named")
    
    return dfSp_Ren_Ret

# module testing code
if __name__ == '__main__':
#    dfSASP = prep_dfSASP()
#    commit_to_csv(dfSASP, "dfSASP.csv")
    #print(dfSASP.head(20))
    
    dfSp_Ren_Ret = prep_dfSpRenRet()
    commit_to_csv(dfSp_Ren_Ret, "dfSp_Ren_Ret.csv")
    #print(dfSp_Ren_Ret.head(20))
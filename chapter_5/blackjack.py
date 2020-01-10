from datetime import timedelta
import numpy as np
import pandas as pd
import time

from Blackjack.card import Card
from Blackjack.constants import MAX_CARD_SUM, MIN_CARD_SUM
from Blackjack import game
from Blackjack import mc_control, mc_init, mc_prediction
from Blackjack import plot


def compute_prediction(num_episodes: int, episodes_from_disk=True, v_from_disk=True):
    assert not(episodes_from_disk == False and v_from_disk == True)
    
    mci = mc_init.MonteCarloInit()
    mcp = mc_prediction.MonteCarloPrediction()

    # (1) compute or load the episodes with policy HIT20
    if episodes_from_disk == False:
        episodes = mci.compute_episodes(num_episodes, commit_to_disk=True)  
    else:
        episodes = mci.load_episodes()  

    # (2) estimate or load the state value function for the episodes
    if v_from_disk == False:
        v = mcp.compute_v(episodes)
    else:
        v = mcp.load_v()

    # plot the value function
    plot_v(v) 
    
def compute_control_ES(num_episodes: int, episodes_from_disk=True, pi_from_disk=True):
    assert not(episodes_from_disk == False and pi_from_disk == True)
    
    mci_equal_probs = mc_init.MonteCarloInit(soft_policy=False, equal_probs=True)
    mcp = mc_prediction.MonteCarloPrediction()
    mcc = mc_control.MonteCarloControl_ES_FirstVisit()

    # (1) load the state value function, initialized at v(s)=0. for all s
    v = mcp.load_v()
    
    # (2) load the policy, initialized at HIT20
    pi = mci_equal_probs.load_pi()
    
    # (3) compute or load the episodes, such that for each state, the taken actions have approx. equal probability
    if episodes_from_disk == False:
        episodes = mci_equal_probs.compute_episodes(num_episodes, commit_to_disk=True)
    else:
        episodes = mci_equal_probs.load_episodes()
        
    # (4) compute or load the optimal policy and action value function for the episodes
    if pi_from_disk == False:
        pi, q = mcc.compute(episodes, pi)
    else:
        pi, q = mcc.load_pi(), mcc.load_q()
        
    # (5) compute the value function from the state value function
    v = mcc.compute_v_from_q(v, q)
    
    # plot the policy
    # TODO
    
    # plot the value function
    plot_v(v)  
    
def compute_control_on_policy(num_episodes: int, episodes_from_disk=True, pi_from_disk=True):
    assert not(episodes_from_disk == False and pi_from_disk == True)
    
    mci = mc_init.MonteCarloInit(soft_policy=True, equal_probs=False)
    mcp = mc_prediction.MonteCarloPrediction()
    mcc = mc_control.MonteCarloControl_OnP_FirstVisit()

    # (1) load the state value function, initialized at v(s)=0. for all s
    v = mcp.load_v()
    
    # (2) load the stochastic policy, initialized at epsilon-soft HIT20
    pi = mci.load_pi()
    
    # (3) compute or load the episodes, such that for each state, the taken actions have approx. equal probability
    if episodes_from_disk == False:
        episodes = mci.compute_episodes(num_episodes, commit_to_disk=True)
    else:
        episodes = mci.load_episodes()
        
    # (4) compute or load the optimal policy and action value function for the episodes
    if pi_from_disk == False:
        pi, q = mcc.compute(episodes, pi)
    else:
        pi, q = mcc.load_pi(), mcc.load_q()
        
    # (5) compute the value function from the state value function
    v = mcc.compute_v_from_q(v, q)
    
    # plot the policy
    # TODO
    
    # plot the value function
    plot_v(v)    
    
def plot_v(v):
    # pivot data for has_usable_ace = 1
    q = v[np.ix_(v[:,2].astype(int) == 1, (0,1,3))]
    dfQ = pd.DataFrame(
        data=q[:, :], # values
        index=[i for i in range(1, len(q) + 1)], # new 1st column as index
        columns=["states_k_sum", "states_k_upcard_value", "rewards_k_plus_1"]
    )
    dfQ_pivoted = dfQ.pivot(
        index="states_k_sum", columns="states_k_upcard_value", values="rewards_k_plus_1")
    plot.plot_Q(dfQ_pivoted)

    # pivot data for has_usable_ace = 0
    q = v[np.ix_(v[:,2].astype(int) == 0, (0,1,3))]
    dfQ = pd.DataFrame(
        data=q[:, :], # values
        index=[i for i in range(1, len(q) + 1)], # new 1st column as index
        columns=["states_k_sum", "states_k_upcard_value", "rewards_k_plus_1"]
    )
    dfQ_pivoted = dfQ.pivot(index="states_k_sum", columns="states_k_upcard_value", values="rewards_k_plus_1")
    plot.plot_Q(dfQ_pivoted)
    

if __name__ == "__main__":
    # set the number of episodes (= Blackjack games) to be simulated.
    num_episodes = 500000

    #compute_prediction(num_episodes, episodes_from_disk=False, v_from_disk=False)
    compute_control_on_policy(num_episodes, episodes_from_disk=False, pi_from_disk=False)
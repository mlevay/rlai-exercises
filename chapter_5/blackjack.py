from datetime import timedelta
import numpy as np
import pandas as pd
from progressbar import ProgressBar
import time

from Blackjack.card import Card
from Blackjack.constants import MAX_CARD_SUM, MIN_CARD_SUM, PLAYER_STICKS_AT
from Blackjack import game
from Blackjack import mc_control, mc_init, mc_prediction
from Blackjack import plot


def compute_prediction(num_episodes: int, episodes_from_disk: bool=True, v_from_disk: bool=True):
    assert not(episodes_from_disk == False and v_from_disk == True)
    
    # compute or load the episodes with fixed r/o policy HIT20
    mci = mc_init.MonteCarloInit()
    if episodes_from_disk == True:
        episodes = mci.load_episodes()  
    else:
        pi = mci.init_pi_of_s(PLAYER_STICKS_AT)
        mci.start_compute(commit_to_disk=True)
        episodes = [None]*num_episodes
        pb = ProgressBar().start()
        for i in range(num_episodes):
            episodes[i] = mci.compute_episode(pi)
            pb.update(int(i / num_episodes * 100))
        mci.end_compute()
        pb.update(100)

    # estimate or load the state value function for the episodes
    mcp = mc_prediction.MonteCarloPrediction()
    if v_from_disk == True:
        v = mcp.load_v()
    else:
        v = mcp.compute_v(episodes)

    # plot the value function
    plot_v(v) 
    
def compute_control_ES(num_episodes: int, pi_and_q_from_disk: bool=True):
    mcc = mc_control.MonteCarloControl_ES_FirstVisit()
    if pi_and_q_from_disk == True:
        # load the optimal policy and action value function for the episodes
        pi, q = mcc.load_pi(), mcc.load_q()
    else:
        # load the deterministic policy, initialized at HIT20
        mci = mc_init.MonteCarloInit(exploring_starts=True)
        pi = mci.init_pi_of_s(PLAYER_STICKS_AT)
        
        # compute the episodes with exploring starts    
        mci.start_compute(commit_to_disk=False)
        mcc.start_compute()
        pb = ProgressBar().start()
        for i in range(num_episodes):
            # source an episode
            episode = mci.compute_episode(pi)
            
            # update the optimal policy and action value function for the episode
            pi, q = mcc.compute_episode(episode, pi)
            pb.update(int(i / num_episodes * 100))
        mci.end_compute()
        mcc.end_compute()
        pb.update(100)
            
    # initialize the state value function with v(s)=0. for all s
    mcp = mc_prediction.MonteCarloPrediction()
    v = mcp.init_v()
    
    # compute the state value function from the action value function
    v = mcc.compute_v_from_q(v, q)
    
    # plot the policy
    plot_pi(pi)
    
    # plot the value function
    plot_v(v)  
    
def compute_control_on_policy(num_episodes: int, pi_and_q_from_disk=True):
    mcc = mc_control.MonteCarloControl_OnP_FirstVisit()
    if pi_and_q_from_disk == True:
        # load the optimal policy and action value function for the episodes
        pi, q = mcc.load_pi(), mcc.load_q()
    else:        
        # load the stochastic policy, initialized at epsilon-soft HIT20
        mci = mc_init.MonteCarloInit(exploring_starts=False)
        pi = mci.init_pi_of_s_and_a(PLAYER_STICKS_AT)
        
        # compute the episodes with exploring starts    
        mci.start_compute(commit_to_disk=False)
        mcc.start_compute()
        pb = ProgressBar().start()
        for i in range(num_episodes):
            # source an episode
            episode = mci.compute_episode(pi)
            
            # update the optimal policy and action value function for the episode
            pi, q = mcc.compute_episode(episode, pi)
            pb.update(int(i / num_episodes * 100))
        mci.end_compute()
        mcc.end_compute()
        pb.update(100)
    
    # plot the policy
    plot_pi(pi)
    
def plot_v(v: np.ndarray):
    c_cs, c_uc, c_hua, c_v = 0, 1, 2, 3
    
    # plot data for has_usable_ace = 1
    v1 = v[np.ix_(v[:, c_hua].astype(int) == 1, (c_cs, c_uc, c_v))]
    _plot_v(v1)

    # plot data for has_usable_ace = 0
    v0 = v[np.ix_(v[:, c_hua].astype(int) == 0, (c_cs, c_uc, c_v))]
    _plot_v(v0)
    
def _plot_v(v: np.ndarray):
    dfQ = pd.DataFrame(
        data=v[:, :], # values
        index=[i for i in range(1, len(v) + 1)], # new 1st column as index
        columns=["states_k_sum", "states_k_upcard_value", "rewards_k_plus_1"]
    )
    dfQ_pivoted = dfQ.pivot(
        index="states_k_sum", columns="states_k_upcard_value", values="rewards_k_plus_1")
    plot.plot_Q(dfQ_pivoted)
                  
def plot_pi(pi: np.ndarray):
    if pi.shape[1] == 4: 
        # deterministic policy, pi(s)
        c_cs, c_uc, c_hua, c_a = 0, 1, 2, 3
        
        # plot data for has_usable_ace = 1
        pi1 = pi[np.ix_(pi[:, c_hua].astype(int) == 1, (c_cs, c_uc, c_a))]
        _plot_pi(pi1)
        
        # plot data for has_usable_ace = 0
        pi0 = pi[np.ix_(pi[:, c_hua].astype(int) == 0, (c_cs, c_uc, c_a))]
        _plot_pi(pi0)
    elif pi.shape[1] == 5: 
        # stochastic policy, pi(a|s)
        c_cs, c_uc, c_hua, c_a, c_p = 0, 1, 2, 3, 4
        
        # plot data for has_usable_ace = 1
        pi1 = pi[(pi[c_p] > .5) and (pi[c_hua].astype(int) == 1), (c_cs, c_uc, c_a)]
        assert pi1.shape[0]*4 == pi.shape[0]
        _plot_pi(pi1)
        
        # plot data for has_usable_ace = 0
        pi0 = pi[(pi[c_p] > .5) and (pi[c_hua].astype(int) == 1), (c_cs, c_uc, c_a)]
        assert pi0.shape[0]*4 == pi.shape[0]
        _plot_pi(pi0)
        
def _plot_pi(pi: np.ndarray):
    # columns: c_cs, c_uc, c_a 
    dfPi = pd.DataFrame(
        data=pi[:, :], # values
        index=[i for i in range(1, len(pi) + 1)], # new 1st column as index
        columns=["states_k_sum", "states_k_upcard_value", "actions_k"]
    )
    dfPi_pivoted = dfPi.pivot(
        index="states_k_sum", columns="states_k_upcard_value", values="actions_k")
    plot.plot_Pi(dfPi_pivoted)
    

if __name__ == "__main__":
    # set the number of episodes (= Blackjack games) to be simulated.
    num_episodes = 500000

    compute_prediction(num_episodes, episodes_from_disk=False, v_from_disk=False)
    compute_control_ES(num_episodes, pi_and_q_from_disk=False)
    compute_control_on_policy(num_episodes, pi_and_q_from_disk=False)
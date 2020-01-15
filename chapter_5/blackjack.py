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
from Blackjack import stats as bjstats


def compute_prediction(num_episodes: int, episodes_from_disk: bool=True, stats_from_disk: bool=True):
    assert not(episodes_from_disk == False and v_from_disk == True)
    
    stats = bjstats.MCPredictionStats()
    # compute or load the episodes with fixed r/o policy HIT20
    mci = mc_init.MonteCarloInit(stats)
    if episodes_from_disk == True:
        episodes = mci.load_episodes()  
    else:
        # pi = mci.init_pi_of_s(PLAYER_STICKS_AT)
        mci.start_compute(commit_to_disk=True)
        episodes = [None]*num_episodes
        pb = ProgressBar().start()
        for i in range(num_episodes):
            episodes[i] = mci.compute_episode()
            pb.update(int(i / num_episodes * 100))
        mci.end_compute()
        pb.update(100)

    # estimate or load the state value function for the episodes
    mcp = mc_prediction.MonteCarloPrediction(stats)
    if stats_from_disk == True:
        mcp.load_stats()
    else:
        mcp.compute_v(episodes)

    # plot the value function
    plot_v(v[:, [bjstats.MCPredictionStats.COL_CARD_SUM, 
                 bjstats.MCPredictionStats.COL_UPCARD,
                 bjstats.MCPredictionStats.COL_HAS_USABLE_ACE,
                 bjstats.MCPredictionStats.COL_V_OF_S]]) 
    
def compute_control_ES(num_episodes: int, stats_from_disk: bool=True):
    stats = bjstats.MCControlESStats()
    
    mcc = mc_control.MonteCarloControl_ES_FirstVisit(stats)
    if stats_from_disk == True:
        # load the optimal policy and action value function for the episodes
        stats = mcc.load_stats()
    else:
        # load the deterministic policy, initialized at HIT20
        mci = mc_init.MonteCarloInit(stats)
        # pi = mci.init_pi_of_s(PLAYER_STICKS_AT)
        
        # compute the episodes with exploring starts    
        mci.start_compute(commit_to_disk=False)
        mcc.start_compute()
        pb = ProgressBar().start()
        for i in range(num_episodes):
            # source an episode
            episode = mci.compute_episode()
            
            # update the optimal policy and action value function for the episode
            mcc.compute_episode(episode)
            pb.update(int(i / num_episodes * 100))
        mci.end_compute()
        mcc.end_compute()
        pb.update(100)
            
    # compute the state value function from the action value function
    mcc.compute_v_from_q()
    
    # plot the policy (deterministic)
    plot_pi(mcc.stats.get_pis()[:, [bjstats.MCControlESStats.COL_CARD_SUM,
                   bjstats.MCControlESStats.COL_UPCARD,
                   bjstats.MCControlESStats.COL_HAS_USABLE_ACE,
                   bjstats.MCControlESStats.COL_PI_OF_S]])
    
    # plot the value function
    plot_v(mcc.stats.get_vs()[:, [bjstats.MCControlESStats.COL_CARD_SUM,
                 bjstats.MCControlESStats.COL_UPCARD,
                 bjstats.MCControlESStats.COL_HAS_USABLE_ACE,
                 bjstats.MCControlESStats.COL_V_OF_S]])  
    
def compute_control_on_policy(num_episodes: int, stats_from_disk=True):
    stats = bjstats.MCControlOnPolicyStats()
    
    mcc = mc_control.MonteCarloControl_OnP_FirstVisit(stats)
    if stats_from_disk == True:
        # load the optimal policy and action value function for the episodes
        stats = mcc.load_stats()
    else:        
        # load the stochastic policy, initialized at epsilon-soft HIT20
        mci = mc_init.MonteCarloInit(stats)
        # pi = mci.init_pi_of_s_and_a(PLAYER_STICKS_AT)
        
        # compute the episodes with exploring starts    
        mci.start_compute(commit_to_disk=False)
        mcc.start_compute()
        pb = ProgressBar().start()
        for i in range(num_episodes):
            # source an episode
            episode = mci.compute_episode()
            
            # update the optimal policy and action value function for the episode
            mcc.compute_episode(episode)
            pb.update(int(i / num_episodes * 100))
        mci.end_compute()
        mcc.end_compute()
        pb.update(100)
    
    # plot the policy (stochastic)
    plot_pi(mcc.stats.get_pis()[:, bjstats.MCControlOnPolicyStats.COL_CARD_SUM,
                 bjstats.MCControlOnPolicyStats.COL_UPCARD,
                 bjstats.MCControlOnPolicyStats.COL_HAS_USABLE_ACE,
                 bjstats.MCControlOnPolicyStats.COL_A,
                 bjstats.MCControlOnPolicyStats.COL_PI_OF_S_A])
    
def plot_v(v: np.ndarray):
    v = np.unique(v, axis=0)
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
    pi = np.unique(pi, axis=0)
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

    compute_prediction(num_episodes, episodes_from_disk=True, stats_from_disk=False)
    #compute_control_ES(num_episodes, stats_from_disk=False)
    #compute_control_on_policy(num_episodes, stats_from_disk=False)
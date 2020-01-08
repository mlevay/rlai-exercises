from datetime import timedelta
import numpy as np
import pandas as pd
import time

from Blackjack.card import Card
from Blackjack.constants import MAX_CARD_SUM, MIN_CARD_SUM
from Blackjack import game
from Blackjack import mc_control, mc_init, mc_prediction
from Blackjack import plot


def compute(cards: [], num_episodes: int):
    mci = mc_init.MonteCarloInit()
    mci_equal_probs = mc_init.MonteCarloInit(equal_probs=True)
    mcp = mc_prediction.MonteCarloPrediction()
    mcc = mc_control.MonteCarloControl()

    # (1) compute and load the episodes with policy HIT20
    #episodes = mci.compute_episodes(cards, num_episodes, commit_to_disk=True)  
    #episodes = mci.load_episodes()  

    # (2) estimate the state value function for the episodes
    #v = mcp.compute_v(episodes)
    v = mcp.load_v()

    # plot the value function
    #plot_v(v)
    
    # (3) compute the optimal policy for the episodes
    pi = mci_equal_probs.load_pi() # the HIT20 policy
    episodes = mci_equal_probs.compute_episodes([], num_episodes, commit_to_disk=True)
    pi, q = mcc.compute(episodes, pi)
    v = mcc.compute_v_from_q(v, q, pi)
    
    # plot the policy
    
    # estimate the optimal state value function for the episodes
    v = mcp.compute_v(episodes)
    
    # # plot the value function
    # plot_v(v)    
    
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

    # optionally, provide a number of pre-set card decks to play with.
    cards = []
    # cards = [
    #     [Card.Ten, Card.Four, Card.Queen, Card.Four, Card.Jack],
    #     [Card.Nine, Card.Queen, Card.Nine, Card.Four, Card.Ace, Card.Five],
    #     [Card.Ace, Card.King, Card.Eight, Card.Three],
    #     [Card.Jack, Card.Five, Card.King, Card.Queen, Card.Six],
    #     [Card.Jack, Card.Five, Card.King, Card.Queen, Card.Five, Card.Ace],
    #     [Card.Ace, Card.Eight, Card.King, Card.Queen, Card.Two, Card.Ace, Card.Nine],
    #     [Card.King, Card.Queen, Card.Jack, Card.Five, Card.Six],
    #     [Card.King, Card.Queen, Card.Ace, Card.Eight, Card.Two],
    #     [Card.King, Card.Two, Card.Ace, Card.Six, Card.Eight, Card.Six]
    # ]

    compute(cards, num_episodes)
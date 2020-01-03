from datetime import timedelta
import numpy as np
import pandas as pd
import time

from Blackjack.card import Card
from Blackjack.constants import VERBOSE
from Blackjack import game
from Blackjack import game_probs
from Blackjack import mc_prediction
from Blackjack.playback import Playback, playback
from Blackjack import plot


def play_one_game(my_game: game.Game, cards: []=[]):
    outcome = my_game.play(cards=cards)

    if VERBOSE == True:
        print("Game outcome: {}".format(str(outcome).split(".")[-1]))
        if len(playback.episodes) > 0:
            print(playback.episodes[-1].actors_k)
            print(playback.episodes[-1].states_k_sum)
            print(playback.episodes[-1].states_k_upcard_value)
            print(playback.episodes[-1].states_k_has_usable_ace)
            print(playback.episodes[-1].actions_k)
            print(playback.episodes[-1].rewards_k_plus_1)
        print()

def produce_test_data(pi):
    i, j = 0, 0
    total = 500000
    start_time = time.time()
    cards = []
    
    # cards = [
    #     [Card.Ten, Card.Four, Card.Queen, Card.Four, Card.Jack],
    #     [Card.Nine, Card.Queen, Card.Ace, Card.Queen],
    #     [Card.Queen, Card.Nine, Card.Four, Card.Ace, Card.Five],
    #     [Card.Ace, Card.King, Card.Eight, Card.Three],
    #     [Card.Jack, Card.Five, Card.King, Card.Queen, Card.Six],
    #     [Card.Jack, Card.Five, Card.King, Card.Queen, Card.Five, Card.Ace],
    #     [Card.Ace, Card.Eight, Card.King, Card.Queen, Card.Two, Card.Ace, Card.Nine],
    #     [Card.King, Card.Queen, Card.Jack, Card.Five, Card.Six],
    #     [Card.King, Card.Queen, Card.Jack, Card.Five, Card.Five],
    #     [Card.King, Card.Queen, Card.Ace, Card.Eight, Card.Two],
    #     [Card.King, Card.Two, Card.Ace, Card.Six, Card.Eight, Card.Six]
    # ]
    
    playback.start(pi)
    my_game = game.Game()
    while i < total:
        if len(cards) > j:
            play_one_game(my_game=my_game, cards=cards[j])
            j += 1
        else:
            play_one_game(my_game=my_game)
        if len(playback.episodes[-1].actors_k) > 0: 
            i += 1
    playback.end()
    
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed_time)))

    if VERBOSE == True:
        print("All episodes:")
        for episode in playback.episodes:
            print("Episode:")
            print(episode.actors_k)
            print(episode.states_k_sum)
            print(episode.states_k_upcard_value)
            print(episode.states_k_has_usable_ace)
            print(episode.actions_k)
            print(episode.rewards_k_plus_1)

if __name__ == "__main__":
    #game_probs.source_cards()
    
    mcp = mc_prediction.MonteCarloPrediction()
    pi = mcp.load_pi()
    v = mcp.load_v()
    
    produce_test_data(pi)
    
    episodes = playback.load()
    mcp.compute(episodes)
    
    # pivot data for has_usable_ace = 1
    q = mcp._v[np.ix_(mcp._v[:,2].astype(int) == 1, (0,1,3))]
    dfQ = pd.DataFrame(
        data=q[:, :], # values
        index=[i for i in range(1, len(q) + 1)], # new 1st column as index
        columns=["states_k_sum", "states_k_upcard_value", "rewards_k_plus_1"]
    )
    dfQ_pivoted = dfQ.pivot(
        index="states_k_sum", columns="states_k_upcard_value", values="rewards_k_plus_1")
    plot.plot_Q(dfQ_pivoted)
    
    # pivot data for has_usable_ace = 0
    q = mcp._v[np.ix_(mcp._v[:,2].astype(int) == 0, (0,1,3))]
    dfQ = pd.DataFrame(
        data=q[:, :], # values
        index=[i for i in range(1, len(q) + 1)], # new 1st column as index
        columns=["states_k_sum", "states_k_upcard_value", "rewards_k_plus_1"]
    )
    dfQ_pivoted = dfQ.pivot(index="states_k_sum", columns="states_k_upcard_value", values="rewards_k_plus_1")
    plot.plot_Q(dfQ_pivoted)
    
    
    
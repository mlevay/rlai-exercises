from datetime import timedelta
import time

from Blackjack.card import Card
from Blackjack.constants import VERBOSE
from Blackjack import game
from Blackjack.playback import Playback, playback


def run(cards: []=[]):
    produced_valid_episode = False
    my_game = game.Game()
    outcome = my_game.play(cards=cards)
    
    if len(playback.episodes) > 0:
            produced_valid_episode = True

    if VERBOSE == True:
        print("Game outcome: {}".format(str(outcome).split(".")[-1]))
        if len(playback.episodes) > 0:
            produced_valid_episode = True
            print(playback.episodes[-1].actors_k)
            print(playback.episodes[-1].states_k_sum)
            print(playback.episodes[-1].states_k_showing_card_value)
            print(playback.episodes[-1].states_k_has_usable_ace)
            print(playback.episodes[-1].actions_k)
            print(playback.episodes[-1].rewards_k_plus_1)
        print()
    return produced_valid_episode

if __name__ == "__main__":
    playback.start()
    # run([Card.Six, Card.Three, Card.Three, Card.Eight, Card.Queen])
    # run([Card.Ace, Card.Queen, Card.Queen, Card.Queen])
    i = 0
    total = 500000
    start_time = time.time()
    while i < total - 1:
        if run(): i+=1
    playback.end()
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(timedelta(elapsed_time)))

    if VERBOSE == True:
        print("All episodes:")
        for episode in playback.episodes:
            print("Episode:")
            print(episode.actors_k)
            print(episode.states_k_sum)
            print(episode.states_k_showing_card_value)
            print(episode.states_k_has_usable_ace)
            print(episode.actions_k)
            print(episode.rewards_k_plus_1)
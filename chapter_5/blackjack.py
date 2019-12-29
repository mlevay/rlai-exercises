from Blackjack import game
from Blackjack.playback import Playback, playback


def run():
    my_game = game.Game()
    outcome = my_game.play()
    print("Game outcome: {}".format(str(outcome).split(".")[-1]))
    #for episode in range(0, len(playback.actors_k)):
    print(playback.episodes[-1].actors_k)
    print(playback.episodes[-1].states_k_sum)
    print(playback.episodes[-1].states_k_showing_card_value)
    print(playback.episodes[-1].states_k_has_usable_ace)
    print(playback.episodes[-1].actions_k)
    print(playback.episodes[-1].rewards_k_plus_1)
    print()


if __name__ == "__main__":
    run()
    run()
    run()
    run()
    run()
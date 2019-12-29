from Blackjack.card import Card
from Blackjack import game
from Blackjack.playback import Playback, playback


def run(cards: []=[]):
    my_game = game.Game()
    outcome = my_game.play(cards=cards)
    print("Game outcome: {}".format(str(outcome).split(".")[-1]))
    #for episode in range(0, len(playback.actors_k)):
    if len(playback.episodes) > 0:
        print(playback.episodes[-1].actors_k)
        print(playback.episodes[-1].states_k_sum)
        print(playback.episodes[-1].states_k_showing_card_value)
        print(playback.episodes[-1].states_k_has_usable_ace)
        print(playback.episodes[-1].actions_k)
        print(playback.episodes[-1].rewards_k_plus_1)
    print()


if __name__ == "__main__":
    playback.start()
    # run([Card.Six, Card.Three, Card.Three, Card.Eight, Card.Queen])
    # run([Card.Ace, Card.Queen, Card.Queen, Card.Queen])
    run()
    run()
    run()
    run()
    run()
    playback.end()

    print("All episodes:")
    for episode in playback.episodes:
        print("Episode:")
        print(episode.actors_k)
        print(episode.states_k_sum)
        print(episode.states_k_showing_card_value)
        print(episode.states_k_has_usable_ace)
        print(episode.actions_k)
        print(episode.rewards_k_plus_1)
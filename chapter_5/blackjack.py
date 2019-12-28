from Blackjack import game


def run():
    my_game = game.Game()
    outcome = my_game.play()
    print("Game outcome: {}".format(outcome))
    print()


if __name__ == "__main__":
    run()
    run()
    run()
    run()
    run()
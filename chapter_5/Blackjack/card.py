import enum

from .constants import MIN_CURRENT_SUM, MAX_CURRENT_SUM


class Card(enum.Enum):
    Ace = 1
    Two, Three, Four, Five, Six, Seven, Eight, Nine = 2, 3, 4, 5, 6, 7, 8, 9
    Jack, Queen, King = 10, 11, 12
        
    def card_value(self) -> int:
        if self == Card.Ace:
            return 1 # card value as defined within the range of dealer's showing card
        elif self in [Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight, Card.Nine]:
            return self.value
        else:
            return 10
        
class CardsState(enum.Enum):
    Unchanged = -2
    Busted = -1
    BlackJack = 0
    Safe = 1

class Cards():
    def __init__(self):
        self._cards = []
        self.has_usable_ace = False
        self.showing_card = None
        self.state = CardsState.Safe
        
    def count_value(self) -> int:
        """
        Returns the current card count.
        """
        card_sum = 0
        sum_others = sum(i.card_value() for i in self._cards if i != Card.Ace) # card_value() always returns 1 for Ace
        count_aces = len(list(i for i in self._cards if i == Card.Ace))
        
        if count_aces > 0:
            if sum_others + count_aces + 10 <= MAX_CURRENT_SUM:
                card_sum = sum_others + count_aces + 10
                self.has_usable_ace = True
            else:
                card_sum = sum_others + count_aces
                self.has_usable_ace = False
        else:
            card_sum = sum_others
            self.has_usable_ace = False
        
        #print(" ---> card sum: {}, has usable ace: {}".format(card_sum, self.has_usable_ace))
        return card_sum  
        
    def add(self, card: Card) -> CardsState:
        """
        Adds a card to the actor's cards and re-counts cards.
        """
        self._cards.append(card)
        if self.showing_card == None: self.showing_card = card

        card_sum = self.count_value()
        if card_sum == MAX_CURRENT_SUM: return CardsState.BlackJack
        if card_sum > MAX_CURRENT_SUM: return CardsState.Busted
        return CardsState.Safe
        
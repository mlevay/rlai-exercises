import enum
import random

from .constants import MIN_CARD_SUM, MAX_CARD_SUM


class Card(enum.Enum):
    Ace = 1
    Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten = 2, 3, 4, 5, 6, 7, 8, 9, 10
    Jack, Queen, King = 11, 12, 13
        
    def card_value(self) -> int:
        if self == Card.Ace:
            return 1 # card value as defined within the range of dealer's showing card
        elif self in [Card.Two, Card.Three, Card.Four, Card.Five, Card.Six, Card.Seven, Card.Eight, Card.Nine, Card.Ten]:
            return self.value
        else:
            return 10
    
    @staticmethod
    def get_card_for_value(value):
        assert (value >= 1 and value <= 10)
        if value == 10: return random.choice([c for c in list(Card) if c.card_value() == 10])
        return Card(value)
    
    @staticmethod
    def min_cardvalue(): return 0
    
    @staticmethod
    def max_cardvalue(): return 10
        
class CardsState(enum.Enum):
    Stuck = -2
    Busted = -1
    Safe = 0
    MaxCnt = 1

class Cards():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._cards = []
        self.has_usable_ace = False
        self.upcard = None
        self.state = CardsState.Safe
        
    @staticmethod
    def _count_value(cards) -> (int, bool):
        """
        Returns the card count for the specified cards.
        """
        card_sum, has_usable_ace = 0, False
        sum_others = sum(i.card_value() for i in cards if i != Card.Ace) # card_value() always returns 1 for Ace
        count_aces = len(list(i for i in cards if i == Card.Ace))
        
        if count_aces > 0:
            if sum_others + count_aces + 10 <= MAX_CARD_SUM:
                card_sum = sum_others + count_aces + 10
                has_usable_ace = True
            else:
                card_sum = sum_others + count_aces
                has_usable_ace = False
        else:
            card_sum = sum_others
            has_usable_ace = False
        
        return card_sum, has_usable_ace
        
    def count_value(self) -> int:
        """
        Returns the current card count.
        """
        card_sum, has_usable_ace = Cards._count_value(self._cards)
        self.has_usable_ace = has_usable_ace        
        return card_sum  
        
    def add(self, card: Card) -> CardsState:
        """
        Adds a card to the actor's cards and re-counts cards.
        """
        self._cards.append(card)
        if self.upcard == None: self.upcard = card

        card_sum = self.count_value()
        if card_sum == MAX_CARD_SUM: return CardsState.MaxCnt
        if card_sum > MAX_CARD_SUM: return CardsState.Busted
        else: return CardsState.Safe
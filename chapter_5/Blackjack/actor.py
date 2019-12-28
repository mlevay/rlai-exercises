import enum
import random

from .card import Card, Cards, CardsState
from .common import enum_to_string
from .constants import ACTOR_DEALER, ACTOR_PLAYER


class Action(enum.Enum):
    Stick = 0
    Hit = 1

class Actor():
    def __init__(self):
        self.cards = Cards()        
        self.dealer = None
        
    def __repr__(self):
        return ACTOR_DEALER if isinstance(self, Dealer) else ACTOR_PLAYER
        
    def hit(self) -> CardsState:
        return self.dealer.deal_card(self)
    
    def stick(self) -> CardsState:
        return CardsState.Unchanged
    
    def take_turn(self) -> (Action, CardsState):
        return (Action.Stick, self.stick())
        
class Dealer(Actor):
    def __init__(self):
        super().__init__()
        
    def deal_card(self, actor: Actor, card: Card = None) -> CardsState:
        new_card = random.choice(list(Card)) if card == None else card
        new_card_value = "1/11"
        if new_card != Card.Ace:
            new_card_value = str(new_card.card_value()) 
        print("{} is dealt a card: {} (value = {})".format(actor, enum_to_string(new_card), new_card_value))
        return actor.cards.add(new_card)
    
    def take_turn(self) -> (Action, CardsState):
        if self.cards.count_value() < 17:
            action = Action.Hit  
            print("{} takes action {}.".format(self, enum_to_string(action)))
            result = self.hit()
        else:
            action = Action.Stick 
            print("{} takes action {}.".format(self, enum_to_string(action)))
            result = self.stick()

        return (action, result)
        
class Player(Actor):
    def __init__(self):
        super().__init__()
        self.showing_card = self.cards.showing_card
    
    def take_turn(self) -> (Action, CardsState):
        if self.cards.count_value() < 20:
            action = Action.Hit  
            print("{} takes action {}.".format(self, enum_to_string(action)))
            result = self.hit()
        else:
            action = Action.Stick 
            print("{} takes action {}.".format(self, enum_to_string(action)))
            result = self.stick()

        return (action, result)
        
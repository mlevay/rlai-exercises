import enum
import numpy as np
import random

from .card import Card, Cards, CardsState
from .common import enum_to_string
from .constants import ACTOR_DEALER, ACTOR_PLAYER, DEALER_STICKS_AT, MIN_CURRENT_SUM, VERBOSE


class Action(enum.Enum):
    Stick = 0
    Hit = 1

class Actor():
    def __init__(self):
        self.dealer = None
        self.reset_cards()
        
    def __repr__(self):
        return ACTOR_DEALER if isinstance(self, Dealer) else ACTOR_PLAYER
        
    def hit(self):
        self.set_cards_state(self.dealer.deal_card(self))
    
    def stick(self):
        self.set_cards_state(self.dealer.deal_no_card(self))
        
    def get_cards_state(self):
        return self._cards_state
        
    def set_cards_state(self, cards_state: CardsState):
        self._cards_state = cards_state
    
    def reset_cards(self):
        self.cards = Cards()        
        self._cards_state = CardsState.Safe
        
class Dealer(Actor):
    def __init__(self):
        super().__init__()
        self._deck = []
        
    def set_deck(self, cards: []):
        deck_size = 20
        if cards:
            self._deck = cards
        for i in range(max(0, len(self._deck)), deck_size):
            self._deck.append(random.choice(list(Card)))
            
    def _pop_card(self):
        return self._deck.pop(0)
        
    def deal_card(self, actor: Actor) -> CardsState:
        new_card = self._pop_card()
        if VERBOSE == True:
            new_card_value = "1/11"
            if new_card != Card.Ace:
                new_card_value = str(new_card.card_value()) 
            print("{} <- {} (value = {})".format(str(actor).upper(), enum_to_string(new_card).upper(), new_card_value))
        
        return actor.cards.add(new_card)
    
    def deal_no_card(self, actor: Actor) -> CardsState:
        actor.cards.count_value() # ensure flags like has_usable_ace are set correctly
        return CardsState.Stuck
    
    def take_turn(self) -> Action:
        if self.cards.count_value() < DEALER_STICKS_AT:
            action = Action.Hit  
            if VERBOSE == True:
                print(".. {}.{}()".format(str(self).upper(), enum_to_string(action).upper()))
            self.hit()
        else:
            action = Action.Stick 
            if VERBOSE == True:
                print(".. {}.{}()".format(str(self).upper(), enum_to_string(action).upper()))
            self.stick()

        return action
        
class Player(Actor):
    def __init__(self):
        super().__init__()
        self._policy = np.array([])
    
    def set_policy(self, pi):
        self._policy = pi
    
    def take_turn(self) -> Action:
        assert self._policy.size != 0
        
        p_card_sum = self.cards.count_value()
        d_upcard_value = self.dealer.cards.upcard.card_value()
        p_has_usable_ace = 1 if self.cards.has_usable_ace else 0
        action = Action.Hit if self._policy[
            (self._policy[:,0] == p_card_sum) & \
            (self._policy[:,1] == d_upcard_value) & \
            (self._policy[:,2] == p_has_usable_ace)][0][3] == Action.Hit.value else Action.Stick
            
        if VERBOSE == True:
            print(".. {}.{}()".format(str(self).upper(), enum_to_string(action).upper()))
            
        if action == Action.Hit:
            self.hit()
        else:
            self.stick()

        return action
        
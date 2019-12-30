import enum
import numpy as np
import random

from .card import Card, Cards, CardsState
from .common import enum_to_string
from .constants import ACTOR_DEALER, ACTOR_PLAYER, DEALER_STICKS_AT, MIN_CURRENT_SUM, PLAYER_STICKS_AT, VERBOSE
from .playback import Playback, playback


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
    
    def reset_cards(self):
        self.cards = Cards()
        
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
            print("{} is dealt a card: {} (value = {})".format(actor, enum_to_string(new_card), new_card_value))
        
        result = actor.cards.add(new_card)
        return result
    
    def take_turn(self) -> (Action, CardsState):
        if self.cards.count_value() < DEALER_STICKS_AT:
            action = Action.Hit  
            if VERBOSE == True:
                print("{} takes action {}.".format(self, enum_to_string(action)))
            result = self.hit()
        else:
            action = Action.Stick 
            if VERBOSE == True:
                print("{} takes action {}.".format(self, enum_to_string(action)))
            result = self.stick()

        return (action, result)
        
class Player(Actor):
    def __init__(self):
        super().__init__()
        self._policy = np.array([])
    
    def set_policy(self, pi):
        self._policy = pi
    
    def take_turn(self) -> (Action, CardsState):
        assert self._policy.size != 0
        
        p_card_sum = self.cards.count_value()
        if p_card_sum < MIN_CURRENT_SUM:
            action = Action.Hit
        else:
            d_showing_card_value = self.dealer.cards.showing_card.card_value()
            p_has_usable_ace = 1 if self.cards.has_usable_ace else 0
            action = Action.Hit if self._policy[
                (self._policy[:,0] == p_card_sum) & \
                (self._policy[:,1] == d_showing_card_value) & \
                (self._policy[:,2] == p_has_usable_ace)][0][3] == Action.Hit.value else Action.Stick
            
        if VERBOSE == True:
            print("{} takes action {}.".format(self, enum_to_string(action)))
            
        if action == Action.Hit:
            result = self.hit()
        else:
            result = self.stick()

        return (action, result)
        
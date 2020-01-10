import enum
import numpy as np
import operator
import random

from .card import Card, Cards, CardsState
from .common import enum_to_string
from .constants import ACTOR_DEALER, ACTOR_PLAYER, DEALER_STICKS_AT
from .constants import MAX_CARD_SUM, MIN_CARD_SUM, VERBOSE


class Action(enum.Enum):
    Stick = 0
    Hit = 1

class Actor():
    def __init__(self):
        self.dealer = None
        self.reset_cards()
        
    def __repr__(self):
        if isinstance(self, ESDealer) == True:
            return "ES_" + ACTOR_DEALER
        elif isinstance(self, ESPlayer) == True:
            return "ES_" + ACTOR_PLAYER
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
        
    def set_deck(self):
        deck_size = 20
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
        actions = self._policy[
            (self._policy[:, 0] == p_card_sum) & \
            (self._policy[:, 1] == d_upcard_value) & \
            (self._policy[:, 2] == p_has_usable_ace)]
        if len(actions) == 1: # deterministic policy pi(s)
            action = Action.Hit if actions[0][3] == Action.Hit.value else Action.Stick
        else: # stochastic policy pi(a|s)
            action = Action(actions[0, 3]) if actions[0, 4] > actions[1, 4] else Action(actions[1, 3])
            
        if VERBOSE == True:
            print(".. {}.{}()".format(str(self).upper(), enum_to_string(action).upper()))
            
        if action == Action.Hit:
            self.hit()
        else:
            self.stick()

        return action        
           
class Tracker():
    def __init__(self):
        self.stats = self._init_stats()
    
    def _init_stats(self):
        all_card_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM + 1))
        all_upcards = [c.value for c in Card]
        all_usable_ace_states = [0, 1]
        all_actions = [a.value for a in Action]
        
        params = np.array(np.meshgrid(
            all_card_sums, all_upcards, all_usable_ace_states, all_actions)).T.reshape(-1, 4).tolist()
        stats = np.zeros((len(params), 5), dtype=int)
        stats[:, :-1] = params
        return stats
    
class ESActor(Actor):
    def __init__(self, stats):
        Actor.__init__(self)
        self.stats = stats
    
    def get_stats_for_state(
        self, card_sum: int, upcard: Card, has_usable_ace: bool) -> np.ndarray:
        upcard, has_usable_ace = upcard.value, int(has_usable_ace)
        stats = self.stats[
            (self.stats[:, 0] == card_sum) & \
            (self.stats[:, 1] == upcard) & \
            (self.stats[:, 2] == has_usable_ace)
        ]
        return stats
    
class ESDealer(ESActor, Dealer):
    def __init__(self, stats):
        ESActor.__init__(self, stats)
        Dealer.__init__(self)
        
    def deal_init_cards(self, player) -> (CardsState, CardsState, Action):
        all_cards = [c for c in list(Card)]
        all_cards_but_ace = [c for c in all_cards if c != Card.Ace]
        
        min_index = np.argmin(self.stats[:, 4], axis=0)
        min_count = self.stats[min_index, 4]
        stats = self.stats[self.stats[:, 4] == min_count, :]
        min_s_a = random.sample(list(stats), 1)[0]
        
        card_sum = min_s_a[0]
        upcard = Card.get_card_for_value(min(min_s_a[1], 10))
        has_usable_ace = bool(min_s_a[2])
        action = Action(min_s_a[3])
            
        self.cards.add(upcard)
        dealer_cs = self.cards.add(random.choice(all_cards))
        
        if has_usable_ace == True:
            player.cards.add(Card.Ace)
            rest = card_sum - 11 # 1 <= rest <= 10
            player_cs = player.cards.add(Card.get_card_for_value(rest))
        else:
            player.cards.add(Card.get_card_for_value(10))
            rest = card_sum - 10 # 2 <= rest <= 11
            if rest >= 10:
                player.cards.add(Card.get_card_for_value(8))
                player_cs = player.cards.add(Card.get_card_for_value(rest - 8))
            else:
                player_cs = player.cards.add(Card.get_card_for_value(rest))
        assert player.cards.count_value() == card_sum
        
        # increment the visits counter for this state and action
        self.stats[
            (self.stats[:, 0] == card_sum) & \
            (self.stats[:, 1] == upcard.value) & \
            (self.stats[:, 2] == int(has_usable_ace)) & \
            (self.stats[:, 3] == action.value), 4
            ] += 1
        
        return dealer_cs, player_cs, action
        
    def deal_card(self, actor: Actor) -> CardsState:
        all_cards = [c for c in list(Card)]
        new_card = random.choice(all_cards)
        
        if VERBOSE == True:
            new_card_value = "1/11"
            if new_card != Card.Ace:
                new_card_value = str(new_card.card_value()) 
            print("{} <- {} (value = {})".format(str(actor).upper(), enum_to_string(new_card).upper(), new_card_value))
        
        return actor.cards.add(new_card)
        
class ESPlayer(ESActor, Player):
    def __init__(self, stats):
        ESActor.__init__(self, stats)
        Player.__init__(self)
        self.first_action = None
    
    def reset_cards(self):
        Player.reset_cards(self)
        
    def set_first_action(self, action: Action):
        self.first_action = action
                        
    def take_turn(self, is_first_turn: bool=False) -> Action:
        assert self._policy.size != 0
        
        if is_first_turn == True:
            # only for the first (s, a) pair after _init(), we must ensure 
            # prob(s, a) is equal for all (s, a)
            action = self.first_action
        else:
            p_card_sum = self.cards.count_value()
            d_upcard_value = self.dealer.cards.upcard.card_value()
            p_has_usable_ace = 1 if self.cards.has_usable_ace else 0
            actions = self._policy[
                (self._policy[:, 0] == p_card_sum) & \
                (self._policy[:, 1] == d_upcard_value) & \
                (self._policy[:, 2] == p_has_usable_ace)]
            if len(actions) == 1: # deterministic policy pi(s)
                action = Action.Hit if actions[0][3] == Action.Hit.value else Action.Stick
            else: # stochastic policy pi(a|s)
                action = Action(actions[0, 3]) if actions[0, 4] > actions[1, 4] else Action(actions[1, 3])
            
        if VERBOSE == True:
            print(".. {}.{}()".format(str(self).upper(), enum_to_string(action).upper()))
            
        if action == Action.Hit:
            self.hit()
        else:
            self.stick()

        return action       
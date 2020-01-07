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
        if isinstance(self, EqualProbabilityDealer) == True:
            return "EP_" + ACTOR_DEALER
        elif isinstance(self, EqualProbabilityPlayer) == True:
            return "EP_" + ACTOR_PLAYER
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
        

class StateActionCounter(object):
    def __init__(self, 
                    card_sum: int = 0, upcard: Card = None, has_usable_ace: bool = False, 
                    action: Action = None):
        self.card_sum = card_sum
        self.upcard = upcard
        self.has_usable_ace = has_usable_ace
        self.action = action
        self.count = 0
        
    @staticmethod
    def init_counters() -> []:
        all_card_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM + 1))
        all_upcards = [c for c in Card]
        all_usable_ace_states = [False, True]
        all_actions = [a for a in Action]
        
        i, stats = 0, [None] * (len(all_card_sums)*len(all_upcards)*len(all_usable_ace_states)*len(all_actions))
        for cs in all_card_sums:
            for uc in all_upcards:
                for hua in all_usable_ace_states:
                    for a in all_actions:
                        stats[i] = StateActionCounter(cs, uc, hua, a)
                        i += 1
        return stats
    
class EqualProbabilityActor(Actor):
    def __init__(self, stats):
        Actor.__init__(self)
        self.stats = stats
    
    def get_stats_for_state(
        self, card_sum: int, upcard: Card, has_usable_ace: bool) -> []:
        stats = list(filter(lambda i: 
                i.card_sum == card_sum and i.upcard == upcard and i.has_usable_ace == has_usable_ace, 
                self.stats))
        return stats
    
class EqualProbabilityDealer(EqualProbabilityActor, Dealer):
    def __init__(self, stats):
        EqualProbabilityActor.__init__(self, stats)
        Dealer.__init__(self)
    
    def _get_least_visited_state_stats(self, cards: [], card_sum: int, upcard: Card, has_usable_ace: bool) -> StateActionCounter:
        # see if it is best to increment the card_sum by 1 (by getting a usable Ace), or get a higher card instead
        all_cards = [c for c in list(Card)]
        all_card_sums = [(0, False)]*len(all_cards)
        for i in range(len(all_cards)):
            new_cards = cards + [all_cards[i]]
            all_card_sums[i] = Cards._count_value(new_cards)
        valid_card_sums = [cs for cs in all_card_sums if cs[0] <= MAX_CARD_SUM]   
        
        if len(valid_card_sums) > 0:
            stats = list(filter(
                lambda item: 
                    (item.card_sum, item.has_usable_ace) in valid_card_sums and \
                    item.upcard == upcard,
                self.stats))
            if len(stats) > 0:
                stats = sorted(stats, key=lambda item: item.count)
                min_count = stats[0].count
                stats = list(filter(lambda item: item.count == min_count, stats))
                return random.choice(stats)
        else: return None
        
    def deal_card(self, actor: Actor) -> CardsState:
        all_cards = [c for c in list(Card)]
        all_cards_but_ace = [c for c in all_cards if c != Card.Ace]
        
        card_sum = actor.cards.count_value()
        if isinstance(actor, EqualProbabilityPlayer) == True:
            if actor.cards.count_value() < MIN_CARD_SUM - 1:
                new_card = random.choice(all_cards)
            else:
                # we need the Player to get an Ace 50% of the time post-_init(); 
                # find the least visited state and action to which the Player can still get from here
                actor_card_sum = actor.cards.count_value() 
                least_visited_sa = self._get_least_visited_state_stats(
                    actor.cards._cards,
                    actor_card_sum,
                    actor.cards.upcard,
                    actor.cards.has_usable_ace)
                # choose the card so that the Player's next state is as found above
                new_card_value = least_visited_sa.card_sum - actor_card_sum 
                if actor.cards.has_usable_ace == True and new_card_value <= 0:
                    new_card_value += 10
                new_card = Card.get_card_for_value(new_card_value)
                actor.set_next_action(least_visited_sa.action)
        else:
            new_card = random.choice(all_cards)
            pass
        
        if VERBOSE == True:
            new_card_value = "1/11"
            if new_card != Card.Ace:
                new_card_value = str(new_card.card_value()) 
            print("{} <- {} (value = {})".format(str(actor).upper(), enum_to_string(new_card).upper(), new_card_value))
        
        return actor.cards.add(new_card)
        
class EqualProbabilityPlayer(EqualProbabilityActor, Player):
    def __init__(self, stats):
        EqualProbabilityActor.__init__(self, stats)
        Player.__init__(self)
        self._next_action = None
    
    def reset_cards(self):
        Player.reset_cards(self)
        self._next_action = None
        
    def set_next_action(self, action: Action):
        self._next_action = action
                        
    def _get_next_action(self, card_sum: int, upcard: Card, has_usable_ace: bool) -> Action:
        assert card_sum >= MIN_CARD_SUM and card_sum <= MAX_CARD_SUM
        card = None
        
        # find the least taken action for the state
        min_count, min_count_sa = 1e6, None
        stats_for_state = self.get_stats_for_state(card_sum, upcard, has_usable_ace)
        assert len(stats_for_state) == 2
        for sa_c in stats_for_state:
            if sa_c.count < min_count:
                min_count, min_count_sa = sa_c.count, sa_c   
        action = min_count_sa.action
        if card_sum == MAX_CARD_SUM: action = Action.Stick
        
        # increment the counter for the current state and chosen action
        min_count_sa.count += 1        
        
        return action
        
    def take_turn(self) -> Action:
        # chooses the next action (and card, if action = Hit) so as to balance the probabilities
        p_card_sum = self.cards.count_value()
        d_upcard_value = Card.get_card_for_value(max(min(self.dealer.cards.upcard.card_value(), Card.max_cardvalue()), Card.min_cardvalue()))
        p_has_usable_ace = True if self.cards.has_usable_ace == 1 else False
        
        action = self._get_next_action(p_card_sum, d_upcard_value, p_has_usable_ace)
            
        if VERBOSE == True:
            print(".. {}.{}()".format(str(self).upper(), enum_to_string(action).upper()))
        
        if action == Action.Hit: 
            self.hit()
        else:
            self.stick()

        return action
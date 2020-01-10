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
        elif isinstance(self, EqualProbabilityPlayer) == True:
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
        self, card_sum: int, upcard: Card, has_usable_ace: bool) -> []:
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
    
    def _get_least_visited_state_stats(self, cards: [], card_sum: int, upcard: Card, has_usable_ace: bool) -> np.ndarray:
        # see if it is best to increment the card_sum by 1 (by getting a usable Ace), or get a higher card instead
        all_cards = [c for c in list(Card)]
        ts = [(0, False)]*len(all_cards)
        for i in range(len(all_cards)):
            new_cards = cards + [all_cards[i]]
            cs, hua = Cards._count_value(new_cards)
            if (cs, int(hua)) not in ts: ts[i] = (cs, int(hua)) 
        valid_ts = [cs for cs in ts if cs[0] > 0 and cs[0] <= MAX_CARD_SUM]   
        
        if len(valid_ts) > 0:
            upcard = upcard.value
            stats = np.zeros((self.stats.shape[0], self.stats.shape[1]+1), dtype=int)
            stats[:, :-1] = self.stats
            for cs, hua in valid_ts:
                indices = np.where(
                    (stats[:, 0] == cs) & (stats[:, 2] == hua) & (stats[:, 1] == upcard)
                )
                if len(indices) > 0: stats[indices, 5] = 1
            stats = stats[stats[:, 5] == 1, :-1]
            if len(stats) > 0:
                min_index = np.argmin(stats[:, 4], axis=0)
                min_count = stats[min_index, 4]
                stats = stats[stats[:, 4] == min_count, :]
                return random.sample(list(stats), 1)[0]
        else: return None
        
    def deal_card(self, actor: Actor) -> CardsState:
        all_cards = [c for c in list(Card)]
        all_cards_but_ace = [c for c in all_cards if c != Card.Ace]
        
        card_sum = actor.cards.count_value()
        if isinstance(actor, EqualProbabilityPlayer) == True:
            if actor.cards.count_value() < MIN_CARD_SUM - 1:
                new_card = random.choice(all_cards)
            else:
                # we need the Player to get an Ace approx. 50% of the time post-_init(); 
                # find the least visited state and action to which the Player can still get from here
                actor_card_sum = actor.cards.count_value() 
                least_visited_sa = self._get_least_visited_state_stats(
                    actor.cards._cards,
                    actor_card_sum,
                    actor.cards.upcard,
                    actor.cards.has_usable_ace)
                # choose the card so that the Player's next state is as found above
                new_card_value = least_visited_sa[0] - actor_card_sum 
                if actor.cards.has_usable_ace == True and new_card_value <= 0:
                    new_card_value += 10
                new_card = Card.get_card_for_value(new_card_value)
                actor.set_next_action(least_visited_sa[3])
        else:
            new_card = random.choice(all_cards)
            pass
        
        if VERBOSE == True:
            new_card_value = "1/11"
            if new_card != Card.Ace:
                new_card_value = str(new_card.card_value()) 
            print("{} <- {} (value = {})".format(str(actor).upper(), enum_to_string(new_card).upper(), new_card_value))
        
        return actor.cards.add(new_card)
        
class EqualProbabilityPlayer(ESActor, Player):
    def __init__(self, stats):
        ESActor.__init__(self, stats)
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
            if sa_c[4] < min_count:
                min_count, min_count_sa = sa_c[4], sa_c   
        action = Action(min_count_sa[3])
        if card_sum == MAX_CARD_SUM: action = Action.Stick
        
        # increment the counter for the current state and chosen action
        self.stats[
            (self.stats[:, 0] == min_count_sa[0]) & \
            (self.stats[:, 1] == min_count_sa[1]) & \
            (self.stats[:, 2] == min_count_sa[2]) & \
            (self.stats[:, 3] == min_count_sa[3]), 4] += 1
        
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
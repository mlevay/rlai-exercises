import enum

from .actor import Action, Actor, Dealer, Player
from .card import Card, Cards, CardsState
from .constants import ACTOR_DEALER, ACTOR_PLAYER
from .constants import MAX_CURRENT_SUM, VERBOSE
from .playback import Playback, playback


class GameOutcome(enum.Enum):
    Ongoing = -3
    DealerReachesFullCount = -2
    DealerWins = -1
    Draw = 0
    PlayerReachesFullCount = 1
    PlayerWins = 2

class Game():
    def __init__(self):
        self.dealer = Dealer()
        self.player = Player()
        
        self.dealer.dealer = self.dealer
        self.player.dealer = self.dealer
        
        self.player.set_policy(playback.pi)
        
        self.player_on_turn = True
    
    def _init(self, cards: []) -> (CardsState, CardsState):
        self.dealer.reset_cards()
        self.player.reset_cards()
        self.player_on_turn = True
        
        # deal first 2 cards to dealer
        self.dealer.deal_card(self.dealer)
        dealer_state = self.dealer.deal_card(self.dealer)
        
        # deal first 2 cards to player
        self.dealer.deal_card(self.player)
        player_state = self.dealer.deal_card(self.player) 
        
        return (dealer_state, player_state)
        
    def _actor_takes_turn(self, dealer_state: CardsState, player_state: CardsState) -> (Actor, Action, CardsState, CardsState):
        """Lets the actor take an action.
        Returns (dealer's card state, player's card state)
        """
        # decide who the actor is
        actor = self.dealer if self.player_on_turn == False else self.player
        adversary_state = dealer_state if self.player_on_turn == True else player_state
        
        # let the actor decide on an action and induce an outcome
        action, state = actor.take_turn()
        
        # see if a change in turns should take place
        if action == Action.Stick: self.player_on_turn = (not self.player_on_turn)
        
        # return the outcome as (dealer's card state, player's card state)
        if isinstance(actor, Dealer): 
            return (actor, action, state, adversary_state) 
        else:
            return (actor, action, adversary_state, state) 
        
    # CardsState = [Unchanged, Busted, BlackJack, Safe]
    def _audit(self, dealer_outcome: CardsState, player_outcome: CardsState) -> GameOutcome:
        assert (not (dealer_outcome == CardsState.Busted and player_outcome == CardsState.Busted))
        assert (not (dealer_outcome == CardsState.BlackJackByFullCount and player_outcome == CardsState.Busted))
        assert (not (dealer_outcome == CardsState.Busted and player_outcome == CardsState.BlackJackByFullCount))
        
        if VERBOSE == True:
            print(" -> Dealer state: {} ({}), Player state: {} ({})".format(
                str(dealer_outcome).split(".")[-1], self.dealer.cards.count_value(), 
                str(player_outcome).split(".")[-1], self.player.cards.count_value()))
            
        # the following represent conditions under which count=21 can no longer be reached
        if dealer_outcome == CardsState.Busted: return GameOutcome.PlayerWins
        if player_outcome == CardsState.Busted: return GameOutcome.DealerWins
        if player_outcome == CardsState.BlackJackByFullCount and \
            dealer_outcome == CardsState.BlackJackByFullCount: return GameOutcome.Draw # possible after _init()
        if dealer_outcome == CardsState.Unchanged and self.dealer.cards.count_value() == MAX_CURRENT_SUM:
            return GameOutcome.DealerWins
        if player_outcome == CardsState.Unchanged and self.player.cards.count_value() == MAX_CURRENT_SUM:
            return GameOutcome.PlayerWins
        if dealer_outcome == CardsState.Unchanged and player_outcome == CardsState.Unchanged:
            dealer_count = self.dealer.cards.count_value()
            player_count = self.player.cards.count_value()
            if dealer_count > player_count: return GameOutcome.DealerWins
            if dealer_count < player_count: return GameOutcome.PlayerWins
            else: return GameOutcome.Draw
        # the following represents conditions under which count=21 may still be reached;
        # this includes when exactly one of the two actors has reached 21 already
        if player_outcome == CardsState.BlackJackByFullCount: return GameOutcome.PlayerReachesFullCount
        if dealer_outcome == CardsState.BlackJackByFullCount: return GameOutcome.DealerReachesFullCount
        return GameOutcome.Ongoing
        
    def play(self, cards: []) -> GameOutcome:
        self.dealer.set_deck(cards=cards)
        playback.start_episode()
        
        dealer_state, player_state = self._init(cards)
        game_state = self._audit(dealer_state, player_state)
        if game_state != GameOutcome.Ongoing: # max. 1 actor has reached count=21 in _init()
            if game_state == GameOutcome.DealerReachesFullCount:
                return GameOutcome.DealerWins
            if game_state == GameOutcome.PlayerReachesFullCount:
                return GameOutcome.PlayerWins
            return game_state # end the game
        
        # it's for the player to take an action first
        self.player_on_turn = True
        
        while True:
            # register the actor and state
            playback.register_actor(self.player_on_turn)
            playback.register_state(
                self.player.cards.count_value(), 
                self.dealer.cards.showing_card.card_value(), # if an Ace, card_value() always returns 1
                self.player.cards.has_usable_ace)
        
            actor, action, dealer_state, player_state = self._actor_takes_turn(dealer_state, player_state)
            # register action taken              
            playback.register_action(action.value)
            game_state = self._audit(dealer_state, player_state)

            # register the reward 
            reward = 0
            if game_state == GameOutcome.DealerWins: reward = -1
            elif game_state == GameOutcome.PlayerWins: reward = 1                
            playback.register_reward(reward)
                
            if game_state not in [
                GameOutcome.Ongoing, 
                GameOutcome.DealerReachesFullCount, 
                GameOutcome.PlayerReachesFullCount]:
                break

        playback.end_episode()        
        return game_state
        
        
        
import enum

from .actor import Action, Actor, Dealer, Player
from .card import Card, Cards, CardsState
from .constants import ACTOR_DEALER, ACTOR_PLAYER, VERBOSE
from .playback import Playback, playback


class GameOutcome(enum.Enum):
    Ongoing = -2
    DealerWins = -1
    Draw = 0
    PlayerWins = 1

class Game():
    def __init__(self):
        self.dealer = Dealer()
        self.player = Player()
        
        self.dealer.dealer = self.dealer
        self.player.dealer = self.dealer
        
        self.player_on_turn = True
        
        self._log = False
    
    def _init(self, cards: []) -> (CardsState, CardsState):
        # if prescribed cards were found, ensure they are passed on
        prescribed_cards = [None]*4 if (not cards) else cards[:min(4,len(cards))]

        # deal first 2 cards to dealer
        self.dealer.deal_card(self.dealer, prescribed_cards[0])
        dealer_init_outcome = self.dealer.deal_card(self.dealer, prescribed_cards[1])
        
        # deal first 2 cards to player
        self.dealer.deal_card(self.player, prescribed_cards[2])
        player_init_outcome = self.dealer.deal_card(self.player, prescribed_cards[3]) 
        
        return (dealer_init_outcome, player_init_outcome)
        
    def _actor_takes_turn(self, dealer_state: CardsState, player_state: CardsState, card: Card = None) -> (Actor, Action, CardsState, CardsState):
        """Lets the actor take an action.
        Returns (dealer's card state, player's card state)
        """
        # decide who the actor is
        actor = self.dealer if self.player_on_turn == False else self.player
        adversary_state = dealer_state if self.player_on_turn == True else player_state
        
        # let the actor decide on an action and induce an outcome
        action, state = actor.take_turn(card)
        
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
        assert (not (dealer_outcome == CardsState.BlackJack and player_outcome == CardsState.Busted))
        assert (not (dealer_outcome == CardsState.Busted and player_outcome == CardsState.BlackJack))
        
        if VERBOSE == True:
            print(" -> Dealer state: {} ({}), Player state: {} ({})".format(
                str(dealer_outcome).split(".")[-1], self.dealer.cards.count_value(), 
                str(player_outcome).split(".")[-1], self.player.cards.count_value()))
        if dealer_outcome == CardsState.Busted: return GameOutcome.PlayerWins
        if player_outcome == CardsState.Busted: return GameOutcome.DealerWins
        if dealer_outcome == CardsState.BlackJack and player_outcome == CardsState.BlackJack: return GameOutcome.Draw
        if dealer_outcome == CardsState.BlackJack: return GameOutcome.DealerWins
        if player_outcome == CardsState.BlackJack: return GameOutcome.PlayerWins
        if dealer_outcome == CardsState.Unchanged and player_outcome == CardsState.Unchanged:
            dealer_count = self.dealer.cards.count_value()
            player_count = self.player.cards.count_value()
            if dealer_count > player_count: return GameOutcome.DealerWins
            if dealer_count < player_count: return GameOutcome.PlayerWins
            else: return GameOutcome.Draw
        return GameOutcome.Ongoing
        
    def play(self, cards: []) -> GameOutcome:
        playback.start_episode()
        
        dealer_state, player_state = self._init(cards)
        game_state = self._audit(dealer_state, player_state)
        if game_state != GameOutcome.Ongoing:
            return game_state
        
        # it's for the player to take an action first
        self.player_on_turn = True
        
        card_i = 4
        while True:
            # register the actor and state
            playback.register_actor(self.player_on_turn)
            playback.register_state(
                self.player.cards.count_value(), 
                self.dealer.cards.showing_card.card_value(), # if an Ace, card_value() always returns 1
                self.player.cards.has_usable_ace)
        
            # pass any further available prescribed cards, or None 
            card = None if len(cards) <= card_i else cards[card_i]
            actor, action, dealer_state, player_state = self._actor_takes_turn(dealer_state, player_state, card)
            # register action taken              
            playback.register_action(action.value)
            game_state = self._audit(dealer_state, player_state)

            # register the reward 
            reward = 0
            if game_state == GameOutcome.DealerWins: reward = -1
            elif game_state == GameOutcome.PlayerWins: reward = 1                
            playback.register_reward(reward)
            playback.end_episode()
            
            card_i += 1
                
            if game_state != GameOutcome.Ongoing:
                break
        
        return game_state
        
        
        
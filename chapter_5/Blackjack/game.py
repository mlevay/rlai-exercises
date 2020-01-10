import enum
import numpy as np

from .actor import Action, Actor
from .actor import Dealer, ESDealer, ESPlayer, Player
from .actor import Tracker
from .card import Card, Cards, CardsState
from .constants import ACTOR_DEALER, ACTOR_PLAYER
from .constants import MIN_CARD_SUM, MAX_CARD_SUM, VERBOSE
from .playback import Playback


class GameState(enum.Enum):
    Continues = -3
    DealerWins = -2
    BothStick = -1
    Draw = 0
    PlayerHasMaxCnt = 1
    PlayerWins = 2

class Game():
    def __init__(self, playback: Playback, exploring_starts=False):
        self._playback = playback     
        self.player_on_turn = True
        self._cl = self._init_cl()
        self.stats = Tracker().stats
        
        # initialize the Dealer
        if exploring_starts == True:
            self.dealer = ESDealer(self.stats)
        else:
            self.dealer = Dealer()
        self.dealer.dealer = self.dealer
        # dealer policy is hard-coded (HIT17)
        
        # initialize the Player 
        if exploring_starts == True:
            self.player = ESPlayer(self.stats)
        else:
            self.player = Player()
        self.player.dealer = self.dealer
    
    def _init(self):
        self.dealer.reset_cards()
        self.player.reset_cards()
        self.player_on_turn = True
        
        # deal first 2 cards to dealer
        self.dealer.set_cards_state(self.dealer.deal_card(self.dealer))
        self.dealer.set_cards_state(self.dealer.deal_card(self.dealer))
        
        # deal first 2 cards to player, and keep dealing further cards if needed 
        # until MIN_CARD_SUM is reached
        self.player.set_cards_state(self.dealer.deal_card(self.player))
        self.player.set_cards_state(self.dealer.deal_card(self.player))
        while self.player.cards.count_value() < MIN_CARD_SUM:
            self.player.set_cards_state(self.dealer.deal_card(self.player))
        
    def _actor_takes_turn(self) -> (Actor, Action):
        """Lets the actor take an action.
        Returns (dealer's card state, player's card state)
        """
        # decide who the actor is
        actor = self.dealer if self.player_on_turn == False else self.player
        
        # let the actor decide on an action and induce an outcome
        action = actor.take_turn()
        
        # see if a change in turns should take place
        if action == Action.Stick: self.player_on_turn = (not self.player_on_turn)
        
        return (actor, action) 

    class NextStep(enum.Enum):
        Stop = 0
        GoOn = 1

    def _init_cl(self) -> dict:
        all_states = [CardsState.Safe, CardsState.Busted, CardsState.Stuck, CardsState.MaxCnt]
        cart_prod = np.array(np.meshgrid(all_states, all_states)).T.reshape(-1, 2)
        keys = list(zip(cart_prod[:, 0], cart_prod[:, 1]))
        values = [(None, None, None)] * len(keys)
        
        cl = dict(zip(keys, values))
        
        #   DEALER               PLAYER                  GAME                        After _init()       Later on
        cl[(CardsState.Safe,     CardsState.Safe   )] = (GameState.Continues,        Game.NextStep.GoOn, Game.NextStep.GoOn)
        cl[(CardsState.Safe,     CardsState.Busted )] = (GameState.DealerWins,       None,               Game.NextStep.Stop)
        cl[(CardsState.Safe,     CardsState.Stuck  )] = (GameState.Continues,        None,               Game.NextStep.GoOn)
        cl[(CardsState.Safe,     CardsState.MaxCnt )] = (GameState.PlayerHasMaxCnt,  Game.NextStep.GoOn, Game.NextStep.GoOn)
        cl[(CardsState.Busted,   CardsState.Stuck  )] = (GameState.PlayerWins,       None,               Game.NextStep.Stop)
        cl[(CardsState.Stuck,    CardsState.Stuck  )] = (GameState.BothStick,        None,               Game.NextStep.Stop)
        cl[(CardsState.MaxCnt,   CardsState.Safe   )] = (GameState.DealerWins,       Game.NextStep.GoOn, None)
        cl[(CardsState.MaxCnt,   CardsState.Stuck  )] = (GameState.DealerWins,       None,               Game.NextStep.Stop)
        cl[(CardsState.MaxCnt,   CardsState.MaxCnt )] = (GameState.Draw,             Game.NextStep.GoOn, None)
        
        return cl

    # CardsState = [Unchanged, Busted, MaxCnt, Safe]
    def _compute(self, is_after_init: bool) -> (int, GameState, NextStep):
        d_state, p_state = self.dealer.get_cards_state(), self.player.get_cards_state()
        d_card_value, p_card_value = self.dealer.cards.count_value(), self.player.cards.count_value()
        g_state = self._cl[(d_state, p_state)]
        what_next = g_state[1] if is_after_init == True else g_state[2]
        
        assert g_state != (None, None, None)
        assert what_next != None
        
        if VERBOSE == True:
            print(" -> {}: {} ({}), {}: {} ({})".format(
                ACTOR_DEALER.upper(), str(d_state).split(".")[-1].upper(), d_card_value, 
                ACTOR_PLAYER.upper(), str(p_state).split(".")[-1].upper(), p_card_value))
            
        reward = 0
        if g_state[0] == GameState.BothStick:
            if p_card_value != d_card_value: 
                reward = 1 if p_card_value > d_card_value else -1
        elif g_state[0] == GameState.DealerWins: reward = -1
        elif g_state[0] == GameState.PlayerWins: reward = 1                
                    
        return reward, g_state[0], what_next
        
    def play(self, pi: np.ndarray) -> (GameState, Playback.Episode):
        self.dealer.set_deck()
        self.player.set_policy(pi)
        self._playback.start_episode()
        self._init()
        
        reward, game_state, what_next = self._compute(True)
        if what_next == Game.NextStep.Stop:
            self._playback.end_episode()
            return game_state, self._playback.episodes[-1]
        
        # it's for the player to take an action first
        self.player_on_turn = True
        
        while True:
            # the player will keep hitting as long as their last action was Hit and their card count is <= 21
            
            prev_player_on_turn = self.player_on_turn
            # register the actor and the state
            self._playback.register_actor(self.player_on_turn)
            self._playback.register_state(
                self.player.cards.count_value(), 
                self.dealer.cards.upcard.card_value(), # if an Ace, card_value() always returns 1
                self.player.cards.has_usable_ace)
        
            actor, action = self._actor_takes_turn()
            # register the action taken              
            self._playback.register_action(action.value)
            
            if prev_player_on_turn == True and game_state != GameState.Continues:
                # we are seeking to exit after init and an extra player.Stick
                
                # register the reward and end the game
                if game_state == GameState.DealerWins: reward = -1
                elif game_state == GameState.PlayerHasMaxCnt: reward = 1                
                self._playback.register_reward(reward)
                break
            
            # advance the game state
            reward, game_state, what_next = self._compute(False)

            # register the reward 
            self._playback.register_reward(reward)
                
            if what_next == Game.NextStep.Stop:
                break

        self._playback.end_episode()        
        return game_state, self._playback.episodes[-1]
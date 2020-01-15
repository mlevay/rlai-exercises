from abc import ABCMeta, abstractmethod
import functools
import numpy as np
import random
from typing import Any, List, NewType, Tuple, Type, TypeVar, Union

from .action import Action
from .card import Card
from .constants import EPSILON, MIN_CARD_SUM, MAX_CARD_SUM, PLAYER_STICKS_AT


T = TypeVar("T", int, float)
State = Tuple[int, int, int]

class Stats(object, metaclass=ABCMeta):
    COL_CARD_SUM, COL_UPCARD, COL_HAS_USABLE_ACE = 0, 1, 2
    _default_key = [-1, -1, -1, -1] # [card_sum, upcard, has_usable_ace, action]
    
    def __init__(self):
        self._cols = None
        self._stats = np.array([])
        
        self._key = []
        self._indices = np.array([])
        self._rows = np.array([])
        
    def _init_cache(self):
        self._key = Stats._default_key.copy()
        self._indices = list(range(self._stats.shape[0]))
        self._rows = self._stats
        
    def _update_cache(self, new_state: State, new_action: Union[Action, int]=None, 
                              column_a: int=-1):
        assert self._stats.size > 0
        
        are_equal = True
        if new_state == None: new_state = list(Stats._default_key).copy()[:3]
        for i in list(range(len(new_state))):
            s_comp = new_state[i]
            if s_comp == None: 
                new_state[i] = list(Stats._default_key).copy()[i]
            if self._key[i] != s_comp:
                self._key[i] = s_comp
                are_equal = False
        if new_action == None: new_action = list(Stats._default_key).copy()[3]
        elif isinstance(new_action, Action): new_action = new_action.value
        if self._key[3] != new_action:
            self._key[3] = new_action
            are_equal = False
        
        if are_equal == True: 
            return # key, values remain the same 

        cond_cols = self._cols[:3] + [column_a]
        count = self._stats.shape[0]
        conditions = np.array([[True]*len(cond_cols)]*count)

        # new_states = np.array([list(new_state)]*count)
        # new_actions = np.array([new_action]*count)
        for i in cond_cols[:-1]:
            if self._key[i] != -1:
                # conditions[:, i] = self._stats[:, i] == new_states[:, i]
                conditions[:, i] = self._stats[:, i] == new_state[i]
        if self._key[-1] != -1:
            # conditions[:, 3] = self._stats[:, column_a] == new_actions[:]
            conditions[:, 3] = self._stats[:, column_a] == new_action
            
        indices = self._get_logical_and_indices(conditions)
        assert len(indices) > 0
        
        self._indices = np.atleast_2d(np.array(indices))
        if len(indices) == 1: 
            self._set_rows(self._stats[indices[0], :])
        else:
            self._set_rows(self._stats[indices][:, :])
                   
    def _get_logical_and_indices(self, conditions) -> List:
        rows = len(conditions)
        columns = 0 if rows == 0 else len(conditions[0])
        result = np.full((rows, 1), True, dtype=bool)
        for i in list(range(columns)):
            result[:, 0] = result[:, 0] & np.array(conditions)[:, i]
        return np.argwhere(result[:, 0] == True)[:, 0].tolist()
        
    def _resolve_state_and_action(self, card_sum: int=None, upcard: Union[Card, int]=None, has_usable_ace: Union[bool, int]=None, 
                  action: Union[Action, int]=None) -> (State, Action):
        s = list(Stats._default_key).copy()[:3]
        if card_sum != None: s[0] = card_sum
        if upcard != None: 
            if isinstance(upcard, Card): upcard = upcard.card_value()
            s[1] = upcard
        if has_usable_ace != None:
            if isinstance(has_usable_ace, bool): has_usable_ace = int(has_usable_ace)
            s[2] = has_usable_ace
        a = list(Stats._default_key).copy()[3]
        if action != None:
            if isinstance(action, Action): action = action.value
            a = action
            
        return tuple(s), a
        
    def _unpack_scalar(self, value: np.ndarray, dupl=False) -> T:
        v: np.ndarray = np.atleast_2d(value)
        if dupl == True:
            assert v.shape in [(1, 2), (2, 1)]
        else:
            assert v.shape == (1, 1)
            
        v = v[0, 0]
        return v
    
    def _unpack_list(self, value: np.ndarray, dupl=False) -> List[T]:
        v: np.ndarray = np.atleast_2d(value)
        assert (v.shape[0] == 1 and v.shape[1] > 1) or (v.shape[0] > 1 and v.shape[1] == 1)
        
        if dupl == True:
            assert v.shape[0] % 2 == 0 or v.shape[1] % 2 == 0
            
            if v.shape[0] == 1: v = v[0, list(range(0, v.shape[1], 2))]
            elif v.shape[1] == 1: v = v[list(range(0, v.shape[0], 2)), 0]
            return v
        else:
            return v.tolist()
    
    @abstractmethod
    def get_stats(self, card_sum: int, upcard: Card, has_usable_ace: bool, action: Action=None) -> np.ndarray:
        return None
        
    def _get_stats(self, state: State = None, action: Union[Action, int] = None, 
                   column_a: int = -1) -> np.ndarray:
        """
        Returns the full row(s) for current state or state-action pair, 
        or alternatively for all rows.
        It doesn't do any duplicate filtering.
        """
        assert not(action != None and column_a < 0)
        
        self._update_cache(state, new_action=action, column_a=column_a)
        return self._rows
        
    def _set_stats(self, values: Union[T, List[T], np.ndarray], cols_to_set: Union[List[int], int]=None, 
                   state: State=None, action: Union[Action, int]=None, column_a: int=-1):
        """
        Sets columns in the rows for a state or state-action pair, or 
        alternatively for all rows.
        Important to be careful with overwriting too many columns - should be limited to 
        a sub-set of column indices by specifying them in cols_to_set. 
        It creates duplicate values without checking.
        """
        if cols_to_set != None: 
            if isinstance(cols_to_set, int) == True: cols_to_set = [cols_to_set]
        else: cols_to_set = self._cols
        
        assert not(state == None and action != None)
        
        self._update_cache(state, new_action=action, column_a=column_a)
        
        # if self._key == Stats._default_key:
        #     self._stats = values
        # else:
        #     if action == None:
        #         indices = np.argwhere(
        #             (self._stats[:, Stats.COL_CARD_SUM].astype(int) == state[0]) & \
        #             (self._stats[:, Stats.COL_UPCARD].astype(int) == state[1]) & \
        #             (self._stats[:, Stats.COL_HAS_USABLE_ACE].astype(int) == state[2]))[:, 0]
        #     else:
        #         if isinstance(action, Action): action = action.value
        #         indices = np.argwhere(
        #             (self._stats[:, Stats.COL_CARD_SUM].astype(int) == state[0]) & \
        #             (self._stats[:, Stats.COL_UPCARD].astype(int) == state[1]) & \
        #             (self._stats[:, Stats.COL_HAS_USABLE_ACE].astype(int) == state[2]) & \
        #             (self._stats[:, column_a].astype(int) == action))[:, 0]
                
        #     rows = self._stats[indices]
        rows = np.atleast_2d(self._rows)
        values = self._reshape_rows(values, (len(rows), len(cols_to_set)))
                
        rows[:, cols_to_set] = values
        self._stats[self._indices, :] = rows
        self._update_cache(state, new_action=action, column_a=column_a)
        
    def _reshape_rows(self, values: Union[T, List[T], np.ndarray], 
                        target_shape: (int, int)) -> np.ndarray:
        target_rows, target_cols = target_shape
        if isinstance(values, List) == False: values = [values] 
        values = np.array(values)
        if len(values.shape) == 1: # 1d array
            assert target_cols == 1 or target_rows in [1, 2]
            assert values.shape[0] == target_cols or values.shape[0] == target_rows or \
                values.shape[0]*2 == target_rows
            
            if values.shape[0] != target_cols:
                if values.shape[0]*2 == target_rows: 
                    values = np.repeat(values, [2], axis=0) # (1,) -> (1, 1), [row] -> [[row],[row]]
                else: 
                    values = np.array([values]).T # (1,) -> (1, 1), [m, m, m] -> [[m], [m], [m]]]
            else:
                values = np.array([values]) # (1,) -> (1, 1), [m, m, m] -> [[m, m, m]]
        elif len(values.shape) == 2:
            assert (values.shape[0] == target_rows or values.shape[0]*2 == target_rows) and values.shape[1] == target_cols
            
            if values.shape[0]*2 == target_rows:
                values = np.repeat(values, [2], axis=0)
        return values
    
    def _set_rows(self, values):
        self._rows = values
    
    def _init_pi_of_s(self, column_pi: int, player_sticks_at: int=PLAYER_STICKS_AT):
        indices = np.argwhere(self._stats[:, Stats.COL_CARD_SUM] >= player_sticks_at)[:, 0]
        rows = self._stats[indices]
        rows[:, column_pi] = Action.Stick.value
        self._stats[indices] = rows
        
        indices = np.argwhere(self._stats[:, Stats.COL_CARD_SUM] < player_sticks_at)[:, 0]
        rows = self._stats[indices]
        rows[:, column_pi] = Action.Hit.value
        self._stats[indices] = rows
    
    def _init_pi_of_s_a_epsilon_soft(self, column_a: int, 
                                     column_pi: int, player_sticks_at: int=PLAYER_STICKS_AT):
        indices = np.argwhere(
            (self._stats[:, Stats.COL_CARD_SUM] >= player_sticks_at) & \
            (self._stats[:, column_a] == Action.Stick.value))[:, 0]
        rows = self._stats[indices]
        rows[:, column_pi] = 1 - EPSILON
        self._stats[indices] = rows
        
        indices = np.argwhere(
            (self._stats[:, Stats.COL_CARD_SUM] >= player_sticks_at) & \
            (self._stats[:, column_a] == Action.Hit.value))[:, 0]
        rows = self._stats[indices]
        rows[:, column_pi] = EPSILON
        self._stats[indices] = rows
        
        indices = np.argwhere(
            (self._stats[:, Stats.COL_CARD_SUM] < player_sticks_at) & \
            (self._stats[:, column_a] == Action.Stick.value))[:, 0]
        rows = self._stats[indices]
        rows[:, column_pi] = EPSILON
        self._stats[indices] = rows
        
        indices = np.argwhere(
            (self._stats[:, Stats.COL_CARD_SUM] < player_sticks_at) & \
            (self._stats[:, column_a] == Action.Hit.value))[:, 0]
        rows = self._stats[indices]
        rows[:, column_pi] = 1 - EPSILON
        self._stats[indices] = rows
        
    def _get_v(self, state: State, column_v: int, dupl: bool=False) -> float:
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None)
        
        return self._unpack_scalar(self._get_stats(state)[column_v], dupl=dupl)
    
    def _set_v(self, state: State, column_v: int, value: float, dupl=False):
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None or value == None)
        
        if dupl == True: value = [value, value]
        self._set_stats(value, column_v, state=state)
        
    def _get_q(self, state: State, action: Union[Action, int], column_a: int, column_q: int) -> float:
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None or action == None)
        
        return self._unpack_scalar(self._get_stats(state, action, column_a=column_a)[column_q])
        
    def _set_q(self, state: State, action: Union[Action, int], column_a: int, column_q: int, value: float):
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None or action == None or value == None)
        
        if isinstance(action, Action): action = action.value
        self._set_stats(value, column_q, state=state, action=action, column_a=column_a)
        
    def _get_pi(self, state: State, column_pi: int, action: Union[Action, int]=None, 
                column_a: int=-1, dupl: bool=False) -> Union[Action, float]:
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None)
        assert not(action != None and column_a < 0)

        pi = self._unpack_scalar(
            self._get_stats(state, action, column_a=column_a)[:, column_pi], 
            dupl=dupl)
            
        if action == None: pi = Action(int(action))
        return pi
        
    def _set_pi(self, state: State, column_pi: int, value: Union[Action, float], 
                action: Union[Action, int]=None, column_a: int=-1, 
                dupl: bool=False) -> Union[Action, float]:
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None or value == None)
        assert not(action != None and column_a < 0)

        if dupl == True: value = [value, value]
        if action != None and isinstance(action, Action): action = action.value
        self._set_stats(value, column_pi, state=state, action=action, column_a=column_a)
            
    def _get_visit_count(self, state: State, column_visits: int, 
                    action: Union[Action, int], column_a: int) -> int:
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None)

        return int(self._unpack_scalar(
                    self._get_stats(state, action, column_a=column_a)[column_visits]))
                    
    def _increment_visit_count(self, state: State, column_visits: int, 
                    action: Union[Action, int], column_a):
        assert not(state == None or state[0] == None or state[1] == None or state[2] == None)

        incr = 1
        stats = self._get_stats(state, action, column_a=column_a)[column_visits]
        self._set_stats(stats + incr, cols_to_set=column_visits, state=state, action=action, column_a=column_a)
    
class MCPredictionStats(Stats):
    COL_V_OF_S, COL_PI_OF_S, COL_VISITS = 3, 4, 5

    def __init__(self):
        super().__init__()
        self._cols = [Stats.COL_CARD_SUM, Stats.COL_UPCARD, Stats.COL_HAS_USABLE_ACE, 
                      MCPredictionStats.COL_V_OF_S, MCPredictionStats.COL_PI_OF_S, 
                      MCPredictionStats.COL_VISITS]
        self._init_stats()
    
    def _init_stats(self):
        all_card_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM + 1))
        all_upcards = list(range(1, 11))
        all_has_usable_ace_states = [0, 1]
        
        self._stats = np.zeros(
            (len(all_card_sums)*len(all_upcards)*len(all_has_usable_ace_states), 6), 
            dtype=float)
        self._stats[:, :3] = np.array(np.meshgrid(
            all_card_sums, all_upcards, all_has_usable_ace_states)).T.reshape(-1, 3)
        
        # intialize pi(s) with HIT21
        self._init_pi_of_s(MCPredictionStats.COL_PI_OF_S)
        
        # initialize the local cache
        self._init_cache()
        
    def get_stats(self, card_sum: int=None, upcard: Union[Card, int]=None, 
                  has_usable_ace: Union[bool, int]=None) -> np.ndarray:
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        return self._get_stats(state=s)
    
    def get_vs(self):
        return self._get_stats()[:, [
            Stats.COL_CARD_SUM, Stats.COL_UPCARD, Stats.COL_HAS_USABLE_ACE, 
            MCPredictionStats.COL_V_OF_S]]
    
    def get_v(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int]) -> float:
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        return self._get_v(s, MCPredictionStats.COL_V_OF_S)
    
    def set_v(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], value: float):
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        self._set_v(s, MCPredictionStats.COL_V_OF_S, value)
        
    def get_visit_count(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int]):
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        return self._get_visit_count(s, MCPredictionStats.COL_VISITS, None, -1)
    
    def increment_visit_count(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int]):
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        self._increment_visit_count(s, MCPredictionStats.COL_VISITS, None, -1)
        
class MCControlESStats(Stats):
    COL_A, COL_Q_OF_S_A, COL_PI_OF_S = 3, 4, 5
    COL_VISITS, COL_START_VISITS = 6, 7
    COL_V_OF_S, COL_PROB = 8, 9
    
    def __init__(self):
        super().__init__()
        self._cols = [Stats.COL_CARD_SUM, Stats.COL_UPCARD, Stats.COL_HAS_USABLE_ACE, 
                      MCControlESStats.COL_A, MCControlESStats.COL_Q_OF_S_A, MCControlESStats.COL_PI_OF_S, 
                      MCControlESStats.COL_VISITS, MCControlESStats.COL_START_VISITS,
                      MCControlESStats.COL_V_OF_S, MCControlESStats.COL_PROB]
        self._init_stats()
        
    def _init_stats(self):
        all_card_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM + 1))
        all_upcards = list(range(1, 11))
        all_has_usable_ace_states = [0, 1]
        all_actions = [Action.Stick.value, Action.Hit.value]
        
        self._stats = np.zeros(
            (len(all_card_sums)*len(all_upcards)*len(all_has_usable_ace_states)*len(all_actions), 10), 
            dtype=float)
        self._stats[:, :4] = np.array(np.meshgrid(
            all_card_sums, all_upcards, all_has_usable_ace_states, all_actions)).T.reshape(-1, 4)
        
        # initialize pi(s) with HIT21
        self._init_pi_of_s(MCControlESStats.COL_PI_OF_S)
        
        # initialize the local cache
        self._init_cache()
        
    def get_stats(self, card_sum: int=None, upcard: Union[Card, int]=None, 
                  has_usable_ace: Union[bool, int]=None, 
                  action: Union[Action, int]=None) -> np.ndarray:
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_stats(state=s, action=a, column_a=MCControlESStats.COL_A)
    
    def get_pis(self):
        return self._get_stats()[:, self._cols[:6]]
        
    def get_vs(self):
        return self._get_stats()[:, self._cols[:9]]
    
    def get_qs(self):
        return self._get_stats()[:, self._cols[7]]
        
    def get_q(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], action: Union[Action, int]) -> float:
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_q(s, a, MCControlESStats.COL_A, MCControlESStats.COL_Q_OF_S_A)
        
    def set_q(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], action: Union[Action, int], value: float):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        self._set_q(s, a, MCControlESStats.COL_A, MCControlESStats.COL_Q_OF_S_A, value)
        
    def get_pi(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int]) -> Action:
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        return self._get_pi(s, MCControlESStats.COL_PI_OF_S, dupl=True)
    
    def set_pi(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], value: Action):
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        self._set_pi(s, MCControlESStats.COL_PI_OF_S, value, dupl=True)
        
    def get_v(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int]) -> float:
        s, _a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        return self._get_v(s, MCControlESStats.COL_V_OF_S, dupl=True)
    
    def set_v(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], value: float):
        s, _ = self._resolve_state_and_action(card_sum, upcard, has_usable_ace)
        self._set_v(s, MCControlESStats.COL_V_OF_S, value, dupl=True)
        
    def get_visit_count(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], action: Union[Action, int]):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_visit_count(s, MCControlESStats.COL_VISITS, a, MCControlESStats.COL_A)
    
    def increment_visit_count(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], action: Union[Action, int]):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        self._increment_visit_count(s, MCControlESStats.COL_VISITS, a, MCControlESStats.COL_A)
        
    def get_start_visits(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], action: Union[Action, int]):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_visit_count(s, MCControlESStats.COL_START_VISITS, a, MCControlESStats.COL_A)
    
    def increment_start_visit_count(self, card_sum: int, upcard: Union[Card, int], 
              has_usable_ace: Union[bool, int], action: Union[Action, int]):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        self._increment_visit_count(s, MCControlESStats.COL_START_VISITS, a, MCControlESStats.COL_A)
    
    def get_state_and_action_with_min_start_visits(self) -> (int, Card, bool, Action):
        min_index = np.argmin(self._stats[:, MCControlESStats.COL_START_VISITS], axis=0)
        min_count = self._stats[min_index, MCControlESStats.COL_START_VISITS]
        stats = self._stats[self._stats[:, MCControlESStats.COL_START_VISITS] == min_count, :]
        min_s_a = random.sample(list(stats), 1)[0]

        card_sum = int(min_s_a[Stats.COL_CARD_SUM])
        upcard = Card.get_card_for_value(min(min_s_a[Stats.COL_UPCARD], 10)) # TODO: check if min is still necessary
        has_usable_ace = bool(min_s_a[Stats.COL_HAS_USABLE_ACE])
        action = Action(min_s_a[MCControlESStats.COL_A])
        
        return card_sum, upcard, has_usable_ace, action
    
    def set_v_and_probs(self, card_sum: int, upcard: Union[Card, int], has_usable_ace: Union[bool, int], 
              prob_stick: float, prob_hit: float, v: float):
        self._set_stats([prob_stick, v], [MCControlESStats.COL_PROB, MCControlESStats.COL_V_OF_S],
                                (card_sum, upcard, has_usable_ace), 
                                action=0, column_a=MCControlESStats.COL_A)
        self._set_stats([prob_hit, v], [MCControlESStats.COL_PROB, MCControlESStats.COL_V_OF_S],
                                (card_sum, upcard, has_usable_ace), 
                                action=1, column_a=MCControlESStats.COL_A)
    
class MCControlOnPolicyStats(Stats):
    COL_A, COL_Q_OF_S_A, COL_PI_OF_S_A, COL_VISITS = 3, 4, 5, 6
    
    def __init__(self):
        super().__init__()
        self._cols = [Stats.COL_CARD_SUM, Stats.COL_UPCARD, Stats.COL_HAS_USABLE_ACE, 
                      MCControlOnPolicyStats.COL_A, MCControlOnPolicyStats.COL_Q_OF_S_A, 
                      MCControlOnPolicyStats.COL_PI_OF_S_A, MCControlOnPolicyStats.COL_VISITS]
        self._init_stats()
        
    def _init_stats(self):
        all_card_sums = list(range(MIN_CARD_SUM, MAX_CARD_SUM + 1))
        all_upcards = list(range(1, 11))
        all_has_usable_ace_states = [0, 1]
        all_actions = [Action.Stick.value, Action.Hit.value]
        
        self._stats = np.zeros(
            (len(all_card_sums)*len(all_upcards)*len(all_has_usable_ace_states)*len(all_actions), 7), 
            dtype=float)
        self._stats[:, :4] = np.array(np.meshgrid(
            all_card_sums, all_upcards, all_usable_ace_states, all_actions)).T.reshape(-1, 4)
        
        # initialize pi(s, a) with epsilon-soft HIT21
        self._init_pi_of_s_a_epsilon_soft(MCControlOnPolicyStats.COL_A, MCControlOnPolicyStats.COL_PI_OF_S)
        
        # initialize the local cache
        self._init_cache()
        
    def get_stats(self, card_sum: int=None, upcard: Union[Card, int]=None, 
                  has_usable_ace: Union[bool, int]=None, 
                  action: Union[Action, int]=None) -> np.ndarray:
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_stats(s, a, MCControlOnPolicyStats.COL_A)
        
    def get_q(self, state: State, action: Union[Action, int]) -> float:
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_q(s, a, MCControlOnPolicyStats.COL_A, MCControlOnPolicyStats.COL_Q_OF_S_A)
        
    def set_q(self, state: State, action: Union[Action, int], value: float):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        self._set_q(s, s, MCControlOnPolicyStats.COL_A, MCControlOnPolicyStats.COL_Q_OF_S_A, value)
        
    def get_pi(self, state: State, action: Union[Action, int]) -> float:
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        return self._get_pi(state, MCControlOnPolicyStats.COL_PI_OF_S_A, action=a, column_a=MCControlOnPolicyStats.COL_A)
    
    def set_pi(self, state: State, value: float, action: Union[Action, int]):
        s, a = self._resolve_state_and_action(card_sum, upcard, has_usable_ace, action=action)
        self.set_pi(state, MCControlOnPolicyStats.COL_PI_OF_S_A, value, action=a, column_a=MCControlOnPolicyStats.COL_A)
        
        
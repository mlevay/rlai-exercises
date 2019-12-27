from .constants import MAX_ACTION, MIN_ACTION
from .state import State


class Action():  
    """
    Represents an action that can be taken in the MDP at any given step.    
    """ 
    @staticmethod
    def is_valid(action_q: int):
        """
        Checks if it is generally valid for a Stake object to be quantified with a given number
        """
        return ((action_q >= MIN_ACTION) and (action_q <= MAX_ACTION))
    
    def __init__(self, action_q: int):
        assert Action.is_valid(action_q)
        
        self.action_q = action_q
        
    # @staticmethod
    # def get_stake(stake: int):
    #     assert (stake >= MIN_ACTION) and (stake <= MAX_ACTION)
        
    #     return Stake(stake)
    
class ValueList():
    """
    Represents the full list of states (capital quantities) and their maximizing policy actions
    """
    def __init__(self):
        states = list(range(MIN_ACTION, MAX_ACTION + 1))        
        self.dict = dict.fromkeys(states, 0)
        
    def get_value(self, state):
        if isinstance(state, int):
            assert State.is_valid(state)
            return self.dict[state]
        elif isinstance(state, State):
            return self.dict[state.state_q]
        else:
            raise TypeError("Function argument is of the wrong type.")
    
    def set_value(self, state, value: float):
        if isinstance(state, int):
            assert State.is_valid(state)
            self.dict[state] = value
        elif isinstance(state, State):
            self.dict[state.state_q] = value
        else:
            raise TypeError("Function argument is of the wrong type.")
    
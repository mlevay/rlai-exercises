from .constants import MAX_STATE, MIN_STATE


class State():
    """
    Represents a state in which the MDP could be at any given step.
    """
    min_valid_state = MIN_STATE
    max_valid_state = MAX_STATE
        
    @staticmethod
    def is_valid(state_q: int):
        """
        Checks if it is generally valid for a Capital object to be quantified with a given number
        """
        return ((state_q >= State.min_valid_state) and (state_q <= State.max_valid_state))        
    
    def __init__(self, state_q: int):
        """
        Initiates a Capital object based on a given capital quantity.
        """
        assert State.is_valid(state_q)
        
        self.state_q = state_q
        self.game_over = (
            (self.state_q == State.min_valid_state) or (self.state_q == State.max_valid_state))
    
class ValueList():
    """
    Represents the full list of states (capital quantities) and their state values
    """
    def __init__(self):
        states = list(range(MIN_STATE, MAX_STATE + 1))        
        self.dict = dict.fromkeys(states, 0.)
        self.dict[len(self.dict) - 1] = 1.
        
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
    
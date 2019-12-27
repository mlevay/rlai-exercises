from .state import State
from .constants import MAX_STATE, MIN_STATE
from .constants import MAX_ACTION, MIN_ACTION
from .action import Action


class SARS():
    """
    Represents the SARS tuple (capital, stake, reward, resulting capital).
    Intuition: with a given capital and valid stake, the CapitalStake
    object will hold one of the 1-2 valid resulting capital as well as
    the respective reward. 
    """
    @staticmethod
    def _get_valid_actions(state):
        """
        Gets all stakes valid for a particular capital quantity
        """
        if isinstance(state, int):
            state = State(state)
        elif isinstance(state, State):
            pass
        else:
            raise TypeError("Function argument is of the wrong type.")
        
        if state.game_over == True:
            return [Action(MIN_ACTION)]
        else:
            return [Action(i) for i in range(MIN_ACTION + 1, min(state.state_q + 1, MAX_STATE - state.state_q + 1))]
        
    @staticmethod
    def _get_resulting_states(state, action):
        if isinstance(state, int):
            state = State(state)
        elif isinstance(state, State):
            pass
        else:
            raise TypeError("Function argument is of the wrong type.")
        
        if isinstance(action, int):
            action = Action(action)
        elif isinstance(action, Action):
            pass 
        else:
            raise TypeError("Function argument is of the wrong type.")
        
        tail_state = State(max(state.state_q - action.action_q, MIN_STATE))
        head_state = State(min(state.state_q + action.action_q, MAX_STATE))
        # if state.state_q >= 85 and state.state_q <= 86: print(
        #     "state: " + str(state.state_q) + \
        #     ", action: " + str(action.action_q) + \
        #     ", tail state: " + str(tail_state.state_q) + \
        #     ", head_state: " + str(head_state.state_q))
        return [tail_state, head_state]
        
    # @staticmethod
    # def get_for_state(state: State):
    #     res = []
    #     for action in SARS._get_valid_actions(state):
    #         res_states = SARS._get_resulting_states(state, action)
    #         for res_state in res_states:
    #             res.append(SARS(state, action, res_state))
    #     return res   
    
    
    def __init__(self, state: State, action: Action, res_state: State):
        self.state = state
        self.action = action
        self.res_state = res_state
        # self.reward = 1. \
        #     if (self.res_state.state_q == State.max_valid_state and self.state.state_q != State.max_valid_state) \
        #     else 0.
        self.reward = 0.
        
    
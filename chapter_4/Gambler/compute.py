from .state import State, ValueList as svl
from .sars import SARS as SARS
from .common import print_status
from .constants import MAX_STATE, MIN_STATE
from .constants import GAMMA, THETA
from .action import Action, ValueList as sal


def value_iteration(prob_heads):
    gamma = GAMMA
    theta = THETA
    
    state_values = svl().dict
    while True:
        delta = 0.
        
        for state in state_values:
            #if (state == MIN_STATE): continue
            state = State(state)
            value = state_values[state.state_q]
            max_value = 0.
            max_action = 0
    
            for action in SARS._get_valid_actions(state):
                action_subtotal = 0.
                res_states = SARS._get_resulting_states(state, action)
                sars0 = SARS(state, action, res_states[0]) # tails outcome
                action_subtotal += ((1. - prob_heads) * (sars0.reward + gamma * state_values[sars0.res_state.state_q]))
                sars1 = SARS(state, action, res_states[1]) # heads outcome
                action_subtotal += (prob_heads * (sars1.reward + gamma * state_values[sars1.res_state.state_q]))

                if action_subtotal > max_value:
                    max_value = action_subtotal
                    max_action = action
            
            state_values[state.state_q] = max_value
            delta = max(delta, abs(value - max_value))
        
        if (delta - theta < 0.): break
        print_status("Values not yet good enough (delta = {}), going for another sweep.".format(delta))
    
    print_status("Values good enough, computing policy.")
        
    s_a_policy = sal().dict
    for state in s_a_policy:
        state = State(state)
        max_value = 0.
        max_action = Action(0)

        for action in SARS._get_valid_actions(state):
            action_subtotal = 0.
            res_states = SARS._get_resulting_states(state, action)
            sars0 = SARS(state, action, res_states[0]) # tails outcome
            action_subtotal += ((1 - prob_heads) * (sars0.reward + gamma * state_values[sars0.res_state.state_q]))
            sars1 = SARS(state, action, res_states[1]) # heads outcome
            action_subtotal += (prob_heads * (sars1.reward + gamma * state_values[sars1.res_state.state_q]))

            action_subtotal = round(action_subtotal, 5)
            if action_subtotal - max_value > 0.: # only prefer higher-q actions if they are significantly higher
                max_value = action_subtotal
                max_action = action
        
        s_a_policy[state.state_q] = max_action.action_q
        
    return state_values, s_a_policy

# module testing code
if __name__ == "__main__":
    pass 
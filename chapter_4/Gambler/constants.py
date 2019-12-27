PROB_HEADS = .4

MIN_STATE = 0 # dummy state, corresponding to termination with capital=0 (loss)
MAX_STATE = 100 # dummy state, corresponding to termination with capital=100 (win)
MIN_ACTION = 0 # dummy action, only available to the terminal states
MAX_ACTION = 99

GAMMA = 1. # undiscounted problem
THETA = 1e-9 # the largest achieved improvement within any single sweep through all states
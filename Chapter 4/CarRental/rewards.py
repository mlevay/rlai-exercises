#from common import *
from constants import *

def compute_transfer_fees(transfer_action):
    assert transfer_action >= -5 and transfer_action <= 5
    if ORIGINAL_PROBLEM == True:
        return abs(transfer_action)*UNIT_COST_OF_TRANSFER
    else:
        if transfer_action > 0: # transfer from A to B
            return max(0, (transfer_action-1)*UNIT_COST_OF_TRANSFER)
        else: # transfer from B to A
            return abs(transfer_action)*UNIT_COST_OF_TRANSFER
        
def compute_parking_fees(next_state_a, next_state_b):
    fees = 0
    if ORIGINAL_PROBLEM == True: return fees
    
    if (next_state_a > 10):
        fees += FEES_PER_PARKING_NIGHT
    if (next_state_b > 10):
        fees += FEES_PER_PARKING_NIGHT
    return fees
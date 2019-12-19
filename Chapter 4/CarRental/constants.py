PATH_SPRENRET_CSV = "C:/Temp/rlai-exercises/Chapter 4/data"
ORIGINAL_PROBLEM = True

EPSILON = 0.01
GAMMA = 0.9
THETA = 0.5

DEFAULT_ACTION = 5
DEFAULT_VALUE = 0.

MIN_NUMBER_OF_CARS_LOC_1 = 0
MIN_NUMBER_OF_CARS_LOC_2 = 0
MAX_NUMBER_OF_CARS_LOC_1 = 20
MAX_NUMBER_OF_CARS_LOC_2 = 20
MAX_NUMBER_OF_CARS_PER_TRANSFER = 5
INDEX_FIRST_CHARGEABLE_TRANSFER = 0
INDEX_LAST_CHARGEABLE_TRANSFER = 4
UNIT_COST_OF_TRANSFER = 2
EXP_VALUE_RENTALS_LOC_1 = 3
EXP_VALUE_RENTALS_LOC_2 = 4
EXP_VALUE_RETURNS_LOC_1 = 3
EXP_VALUE_RETURNS_LOC_2 = 2
REWARD_PER_RENTAL = 10
FEES_PER_PARKING_NIGHT = 4

# Constants for the dataframe dfS_A_Sp
DFCOL_SASP_SORIG = "s_k"
DFCOL_SASP_ACTION = "a_k" # for each original state, a number of actions are valid and possible
DFCOL_SASP_IS_VALID = "is_valid"
DFCOL_SASP_SPSEUDO = "s_pseudo_k" # each sequence (s,a) can result in a number of next states s'
DFCOL_SASP_FEES = "fees_k"
DFCOL_SASP_SORIG_A = "s_k_a"
DFCOL_SASP_SORIG_B = "s_k_b"
DFCOL_SASP_NUM_TRANSFERS = "count_transfers"
DFCOL_SASP_SPSEUDO_A = "s_pseudo_k_a"
DFCOL_SASP_SPSEUDO_B = "s_pseudo_k_b"

# Constants for the dataframe dfSp_ren_ret
DFCOL_SPRENRET_SPSEUDO = "s_pseudo_k"
DFCOL_SPRENRET_RENTALS_A = "rentals_k_a"
DFCOL_SPRENRET_RENTALS_B = "rentals_k_b"
DFCOL_SPRENRET_RETURNS_A = "returns_k_a"
DFCOL_SPRENRET_RETURNS_B = "returns_k_b"
DFCOL_SPRENRET_IS_VALID = "is_valid"
DFCOL_SPRENRET_SNEXT = "s_k_plus_1"
DFCOL_SPRENRET_PROB_RENTALS_A = "p_rentals_a"
DFCOL_SPRENRET_PROB_RENTALS_B = "p_rentals_b"
DFCOL_SPRENRET_PROB_RETURNS_A = "p_returns_a"
DFCOL_SPRENRET_PROB_RETURNS_B = "p_returns_b"
DFCOL_SPRENRET_PROBSRSA = "p_of_srsa" # p(s',r|s,a)
DFCOL_SPRENRET_REWARD = "r_k_reward" # each sequence (s,a,s') results in a unique reward r
DFCOL_SPRENRET_FEES = "fees_k"
DFCOL_SPRENRET_SNEXT_A = "s_k_plus_1_a"
DFCOL_SPRENRET_SNEXT_B = "s_k_plus_1_b"

# Constants for the dataframe dfV
DFCOL_V_STATE = "s" # s
DFCOL_V_STATE_A = "s_a"
DFCOL_V_STATE_B = "s_b"
DFCOL_V_VALUE = "v_of_s" # v(s)

# Constants for the dataframe dfPi
DFCOL_PI_STATE = "s" # s
DFCOL_PI_STATE_A = "s_a"
DFCOL_PI_STATE_B = "s_b"
DFCOL_PI_ACTION = "a" # a
DFCOL_PI_PROB = "p_of_sa"
# whether we're solving the original problem (Example 4.2) 
# or for the additional requirements (Ex. 4.7);
ORIGINAL_PROBLEM = True

# whether to use disk to r/w the following CSV files:
# (-) states, valid next actions and their respective pseudo-
#     states, and the car transfer fees they incur (dfSASP.csv);
# (-) pseudo-states, valid rental/return combinations for them,
#     the next state they lead to, as well as the respective
#     probabilities, rewards and overflow parking fees they
#     incur (dfSp_Ren_Ret.csv);
# (-) states and their respective values, as learned in
#     policy iteration (dfV.csv);
# (-) states and actions, plus their respective probabilities
#     per policy, as learned in policy iteration (dfPi.csv).
# Don't use this option if you don't have >=rw access to disk;
USE_DISK_FOR_CSV_DATA = True

# what directory to use for r/w of CSV files from/to disk
PATH_SPRENRET_CSV = "C:/Temp/rlai-exercises/Chapter 4/data"

# whether to load cached preprocessed data from CSV files
# for purposes of quick visualization w/o a full code run
# (dfSASP.csv, dfSp_Ren_Ret.csv).
# Don't set this to TRUE if USE_DISK_FOR_CSV_DATA = False;
GET_DATA_FROM_CSV = True

# whether to load cached models from CSV files
# for purposes of quick visualization w/o a full code run
# (dfPi.csv, dfV.csv).
# Don't set this to TRUE if USE_DISK_FOR_CSV_DATA = False;
GET_MODEL_FROM_CSV = True

# what file numbers to use for loading the models from
# dfPi.csv, dfV.csv.
# Set both to -1 if GET_MODEL_FROM_CSV = False or to
# the file prefix (int>=0) if GET_MODEL_FROM_CSV = True;
# don't set this to TRUE if USE_DISK_FOR_CSV_DATA = False.
PI_SEQ_NR = 5
V_SEQ_NR = 6

# file name prefixes
FILE_SASP_PREFIX = "dfSASP"
FILE_SPRENRET_PREFIX = "dfSp_Ren_Ret"
FILE_PI_PREFIX = "dfPi"
FILE_V_PREFIX = "dfV"

EPSILON = 0.1
GAMMA = 0.9
THETA = 1.

DEFAULT_ACTION = 5
DEFAULT_VALUE = 0.

MIN_NUMBER_OF_CARS_LOC_1 = 0
MIN_NUMBER_OF_CARS_LOC_2 = 0
MAX_NUMBER_OF_CARS_LOC_1 = 20
MAX_NUMBER_OF_CARS_LOC_2 = 20
MAX_NUMBER_OF_CARS_PER_TRANSFER = 5
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
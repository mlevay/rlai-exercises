import numpy as np
import os
import pandas as pd
import scipy.special

from .constants import MAX_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_2
from .constants import EXP_VALUE_RENTALS_LOC_1, EXP_VALUE_RENTALS_LOC_2
from .constants import EXP_VALUE_RETURNS_LOC_1, EXP_VALUE_RETURNS_LOC_2

def get_probsrsa_vectorized(number, exp_rate):
    """Vectorized Poisson probability computation"""
    return (np.power(exp_rate, number))*(np.exp(-exp_rate))/(scipy.special.factorial(number))

def init_prob_lookup():
    """Prepare a dtype=float lookup table for rental/return Poisson probabilities per expected number and number"""
    poss_numbers = range(max(MAX_NUMBER_OF_CARS_LOC_1, MAX_NUMBER_OF_CARS_LOC_2) + 1)
    exp_numbers = [EXP_VALUE_RENTALS_LOC_1, EXP_VALUE_RENTALS_LOC_2, EXP_VALUE_RETURNS_LOC_1, EXP_VALUE_RETURNS_LOC_2]
        
    # create a cartesian product of expected numbers and rental/return numbers
    prob_lookup = np.array(np.meshgrid(poss_numbers, exp_numbers, [0.])).T.reshape(-1,3)
    prob_lookup[:,2] = get_probsrsa_vectorized(prob_lookup[:,0], prob_lookup[:,1])
    return prob_lookup

prob_lookup = init_prob_lookup()

def lookup_prob(number, exp_number):
    """Look up the probability of _number_ rentals/returns for rate=_exp_number_"""
    result = prob_lookup[:, 2][(prob_lookup[:,0].astype(int) == number) & (prob_lookup[:,1].astype(int) == exp_number)]
    return np.unique(result)

def lookup_prob_vectorized(numbers, exp_number):
    """Vectorized Poisson probability lookup, stretched to ensure sum_prob = 1"""
    _, unique_indices = np.unique(numbers, return_index=True)
    max_indices = np.where(numbers == np.amax(numbers)) # take the index of the largest number
    probs = [0.] * len(numbers)
    for i in range(len(numbers)):
        if not np.isin(i, max_indices):
            probs[i] = lookup_prob(numbers[i], exp_number)
    
    probs_sum_so_far = np.sum(np.take(probs, unique_indices))
    for i in range(len(numbers)):
        if np.isin(i, max_indices):
            probs[i] = 1. - probs_sum_so_far
    return probs
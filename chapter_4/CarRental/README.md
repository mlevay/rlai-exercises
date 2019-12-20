# Exercise 4.7 (programming)

## Question:
Write a program for policy iteration and re-solve <b>Jack's car rental problem</b> with the following changes. One of Jack's employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs \\$2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location. If more than 10 cars are kept overnight at a location (after any moving of cars), than an additional cost of \\$4 must be incurred to use a second parking lot (independent of how many cars are kept there). These sorts of non-linearities and arbitrary dynamics often occur in real problems and cannot easily be handled by optimization methods other than dynamic programming. To check your program, first replicate the results given for the original problem. 

## Answer:
### Discussion and Assumptions:
1. The problem statement does not explicitly mention the use of an <b>$\epsilon$-soft policy</b>. The below code assumes it and sets the value of $\epsilon$ in <b>CarRental/constants.py</b>. Furthermore, our understanding of an $\epsilon$-soft policy is:
 - during <b>Initialization</b> and <b>Policy Improvement</b>, we compute the policy function $\pi(a|s)$, assigning probability $\pi(a_*|s) = 1 - \epsilon + \frac{\epsilon}{|A(s)|}$ for the value-maximizing action $a_*$ of state $s$ and probability $\pi(a_{other}|s) = \frac{\epsilon}{|A(s)|}$ for every other action $a_{other}$ available for state $s$;
 - during <b>Policy Evaluation</b>, we compute the value function $v(s)$ by using an <b>$\epsilon$-greedy policy</b>, which ensures $\epsilon$ probability that any valid action will be selected for each state. This measure of exploration should provide theoretical guarantees for convergence.


2. The Problem statement doesn't mention the value, $\theta$, of the threshold for further refreshes of the value function. The below code sets the value of $\theta$ in <b>CarRental/constants.py</b>.


3. Because the number of rentals/returns at any location is limited to a maximum of 20 cars, the Poisson probabilities of having any number of rentals or returns at any location in a business day don't add up to 1. The below code addresses this by padding the probability of the highest possible number of rentals/returns in any given situation. This way, we theoretically:
 - handle 12 returns at a location with leftover capacity of 3 returns as part of the probability of having only 3 returns that day;
 - handle 12 rentals at a location with only 3 available cars as part of the probability of having rented only 3 cars that day (with no rewards for the extra rental requests that couldn't be fulfilled).
 
 
4. We further assume that, per the extended problem statement (Exercise 4.7), a maximum of 5 cars can be moved overnight from location A to location B (1 shuttled by an employee at no cost and up to 4 for \\$2 a piece). In the original problem (as stated in Example 4.2), every moved car would incur cost. 


5. The code is based on the concept of a <b>pseudo-state</b>, which can be intuitively described as the number of cars across locations at 6am: following transfer of cars but prior to the start of the business day. When in this pseudo-state, which state is reached next depends on Poisson random variables.

### Instructions for running the code:
The code is based on pandas/numpy, and was written with learning/demonstration in mind (as opposed to good practices/performance). 

A few constants can be set below (and have internal pre-sets). The logic itself is stored in the modules chapter_4/CarRental/* and chapter_4/car_rental.py; all constants are in chapter_4/CarRental/constants.py. At the very least, the working directory must be set correctly for the code to run. 

The code can be executed both in IPython (notebook chapter_4/Chapter 4 Exercises.ipynb) and directly (by running the module chapter_4/car_rental.py).

Library versions used: python 3.7.3, numpy 1.15.4, matplotlib 3.1.0, pandas 0.24.2.  
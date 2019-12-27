from Gambler import compute, constants, plot


def run():
    # the following will grab the values either from overrules or from the module itself
    prob_heads = constants.PROB_HEADS
    
    state_values, s_a_policy = compute.value_iteration(prob_heads)
    plot.plot_V(state_values)
    plot.plot_Pi(s_a_policy)
    

# module testing code
if __name__ == "__main__":
    run()
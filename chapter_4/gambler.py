from Gambler import compute, plot


def run():
    state_values, s_a_policy = compute.value_iteration()
    plot.plot_V(state_values)
    plot.plot_Pi(s_a_policy)
    

# module testing code
if __name__ == "__main__":
    run()
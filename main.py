import os
import numpy as np
from Algorithms.WelfaristUCB import simulate_Welfarist_UCB_single
from Algorithms.NCBNonPrivate import simulate_ncb
from Algorithms.ExploreThenUCB import simulate_ExploreUCB_single
from helper import plot_expt_a, plot_expt_b, plot_expt_f


# CHANGE THESE DIRECTORIES
path_to_cached = "/Users/nishantpandey/Desktop/Stuff With Code/UCBisAllYouNeed/Cached"
path_to_results = "/Users/nishantpandey/Desktop/Stuff With Code/UCBisAllYouNeed/Results"

os.makedirs(path_to_cached, exist_ok=True)
os.makedirs(path_to_results, exist_ok=True)

for suffix in ['a', 'b', 'c', 'd', 'e', 'f']:
    subdir = os.path.join(path_to_cached, f"expt_{suffix}")
    os.makedirs(subdir, exist_ok=True)



if __name__ == "__main__":

    np.random.seed(42)

    ##### PARAMETERS
    num_trials = 50
    c = 3.1


    ################# Experiment A: Comparing Nash Regret (p=0) for Standard NCB vs Welfarist UCB for bernoulli rewards #################

    test_type = 0
    T_max = 10000000      # choose your total horizon once
    means = np.random.uniform(0.005, 1, size=50)
    sigma2 = 1

    # nash_regret_standard_ncb = simulate_ncb(means, T_max, sigma2, num_trials, c, test_type, "Nash")
    nash_regret_standard_ncb = np.load("cached/expt_a/nash_regret_standard_NCB.npy")  # load cached regret
    np.save("cached/expt_a/nash_regret_standard_NCB.npy", nash_regret_standard_ncb)

    # nash_regret_welfarist_ucb = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=0)
    nash_regret_welfarist_ucb = np.load("cached/expt_a/nash_regret_welfarist_UCB.npy")  # load cached regret
    np.save("cached/expt_a/nash_regret_welfarist_UCB", nash_regret_welfarist_ucb)

    plot_expt_a([nash_regret_standard_ncb, nash_regret_welfarist_ucb], T_max, test_type)


    ################# Experiment B: Comparing Nash Regret (p=0) for Standard NCB vs Welfarist UCB for subgaussian rewards #################


    T_max = 1000000      
    test_type = 1
    
    means = np.random.uniform(10, 1000, size=50)
    # means[0] = 1e3
    sigma2 = 400

    # nash_regret_standard_ncb = simulate_ncb(means, T_max, sigma2, num_trials, c, test_type, "Nash")
    nash_regret_standard_ncb = np.load("cached/expt_b/nash_regret_NCB.npy")  # load cached regret
    np.save("cached/expt_b/nash_regret_NCB.npy", nash_regret_standard_ncb)

    # nash_regret_welfarist_ucb = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=0)
    nash_regret_welfarist_ucb = np.load("cached/expt_b/nash_regret_UCB_Welfarist.npy")  # load cached regret
    np.save("cached/expt_b/nash_regret_UCB_Welfarist", nash_regret_welfarist_ucb)

    plot_expt_a([nash_regret_standard_ncb, nash_regret_welfarist_ucb], T_max, test_type)



    ####################### Experiment C, D, E: Comparing p-mean regret for Explore-Then-UCB and  Welfarist UCB for different p values #################

    T_max = 1000000      
    test_type = 1
    
    means = np.random.uniform(10, 1000, size=50)
    # means[0] = 1e3
    sigma2 = 400

    ##### p = 0.5 #####
    p = 0.5

    # p_regret_explore_ucb = simulate_ExploreUCB_single(means, T_max, sigma2, num_trials, test_type, p=p)
    p_regret_explore_ucb = np.load("cached/expt_c/p_regret_explore_05.npy")  # load cached regret
    np.save("cached/expt_c/p_regret_explore_05.npy", p_regret_explore_ucb)


    # p_regret_welfarist_ucb = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=p)
    p_regret_welfarist_ucb = np.load("cached/expt_c/p_regret_welfarist_UCB_05.npy")  # load cached regret
    np.save("cached/expt_c/p_regret_welfarist_UCB_05.npy", p_regret_welfarist_ucb)

    plot_expt_b([p_regret_explore_ucb, p_regret_welfarist_ucb], T_max, p)


    p=-0.5

    # p_regret_explore_ucb = simulate_ExploreUCB_single(means, T_max, sigma2, num_trials, test_type, p=p)
    p_regret_explore_ucb = np.load("cached/expt_d/p_regret_explor_m05.npy")  # load cached regret
    np.save("cached/expt_d/p_regret_explor_m05.npy", p_regret_explore_ucb)


    # p_regret_welfarist_ucb = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=p)
    p_regret_welfarist_ucb = np.load("cached/expt_d/p_regret_welfarist_UCB_m05.npy")  # load cached regret
    np.save("cached/expt_d/p_regret_welfarist_UCB_m05", p_regret_welfarist_ucb)

    plot_expt_b([p_regret_explore_ucb, p_regret_welfarist_ucb], T_max, p)


    p=-1.5
    # p_regret_explore_ucb = simulate_ExploreUCB_single(means, T_max, sigma2, num_trials, test_type, p=p)
    p_regret_explore_ucb = np.load("cached/expt_e/p_regret_explor_m15.npy")  # load cached regret
    np.save("cached/expt_e/p_regret_explor_m15.npy", p_regret_explore_ucb)


    # p_regret_welfarist_ucb = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=p)
    p_regret_welfarist_ucb = np.load("cached/expt_e/p_regret_welfarist_UCB_m15.npy")  # load cached regret
    np.save("cached/expt_e/p_regret_welfarist_UCB_m15.npy", p_regret_welfarist_ucb)

    plot_expt_b([p_regret_explore_ucb, p_regret_welfarist_ucb], T_max, p)


######################## Experiment F: Comparing p-mean regret for Welfarist UCB for different p values #################

    T_max = 1000000      
    test_type = 1
    
    means = np.random.uniform(10, 1000, size=50)
    # means[0] = 1e3
    sigma2 = 400

    # p_regret_welfarist_ucb_1 = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=1)
    p_regret_welfarist_ucb_1 = np.load("cached/expt_f/p_regret_welfarist_UCB_1.npy")  # load cached regret
    np.save("cached/expt_f/p_regret_welfarist_UCB_1.npy", p_regret_welfarist_ucb_1)


    # p_regret_welfarist_ucb_0 = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=0)
    p_regret_welfarist_ucb_0 = np.load("cached/expt_f/p_regret_welfarist_UCB_0.npy")  # load cached regret
    np.save("cached/expt_f/p_regret_welfarist_UCB_0.npy", p_regret_welfarist_ucb_0)

    # p_regret_welfarist_ucb_m1 = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=-1)
    p_regret_welfarist_ucb_m1 = np.load("cached/expt_f/p_regret_welfarist_UCB_m1.npy")  # load cached regret
    np.save("cached/expt_f/p_regret_welfarist_UCB_m1.npy", p_regret_welfarist_ucb_m1)


    # p_regret_welfarist_ucb_m2 = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=-2)
    p_regret_welfarist_ucb_m2 = np.load("cached/expt_f/p_regret_welfarist_UCB_m2.npy")  # load cached regret
    np.save("cached/expt_f/p_regret_welfarist_UCB_m2.npy", p_regret_welfarist_ucb_m2)


    # p_regret_welfarist_ucb_m5 = simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p=-5)
    p_regret_welfarist_ucb_m5 = np.load("cached/expt_f/p_regret_welfarist_UCB_m5.npy")  # load cached regret
    np.save("cached/expt_f/p_regret_welfarist_UCB_m5", p_regret_welfarist_ucb_m5)


    plot_expt_f([p_regret_welfarist_ucb_1, p_regret_welfarist_ucb_0, p_regret_welfarist_ucb_m1, p_regret_welfarist_ucb_m2, p_regret_welfarist_ucb_m5], T_max)
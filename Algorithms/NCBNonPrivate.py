import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt


@njit
def clip(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x
    

@njit    
def sample_reward(mean, sigma2,  test_type):
    """
    Sample a reward according to the arm's mean category,
    implemented in pure nopython with only random.random().
    """

    u = np.random.rand()  
    # u = np.random.rand()  
    if test_type == 0:
        return 1.0 if u < mean else 0.0
    
    elif test_type == 1:
        return np.random.normal(mean, np.sqrt(sigma2))
    else:
        if mean >= 10 and mean < 40:  # 50% chance
            return np.random.uniform(mean - np.sqrt(sigma2), mean + np.sqrt(sigma2))
        elif mean >= 40 and mean < 70:
            return np.random.normal(mean, np.sqrt(sigma2))
        else: 
            d = np.sqrt(6.0 * sigma2)     # half-width
            a = mean - d
            b = mean + d
            return np.random.triangular(a, mean, b)




@njit
# def NCB_single(means, c, sigma2, T, test_type):
#     """Run NCB for T steps, return chosen arms[0..T-1]."""
#     k = len(means)
#     n = np.zeros(k, dtype=np.int64)
#     sums = np.zeros(k, dtype=np.float64)
#     mu = np.zeros(k, dtype=np.float64)
#     arms = np.empty(T, dtype=np.int64)

#     logT = np.log(T)
#     t = 0

#     # Phase 1: uniform exploration
#     while t < T and np.max(n * mu) <= 420 * (c**2) * logT:
#         a = np.random.randint(0, k)
#         # r = np.random.random() < means[a] 
#         r = sample_reward(means[a],sigma2,  test_type)

#         n[a] += 1
#         sums[a] += r
#         mu[a] = sums[a] / n[a]
#         arms[t] = a
#         t += 1

#     # Phase 2: NCB selection
#     while t < T:
#         bonus = np.empty(k, dtype=np.float64)
#         for i in range(k):
#             if n[i] == 0:
#                 bonus[i] = 1e12
#             else:
#                 bonus[i] = 2 * c * np.sqrt((2 * mu[i] * logT) / n[i])
#         ncb = mu + bonus
#         A = np.argmax(ncb)
#         r = sample_reward(means[A],sigma2,  test_type)


#         n[A] += 1
#         sums[A] += r
#         mu[A] = sums[A] / n[A]
#         arms[t] = A
#         t += 1

#     return arms
def NCB_single(means, sigma2, c, T, test_type):
    """Run NCB for T steps, return chosen arms[0..T-1]."""
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)

    logT = np.log(T)
    logk = np.log(k)
    t = 0

    # Phase 1: uniform exploration
    # while t < 16 * np.sqrt(k * T * logT/logk):
    while t < T and np.max(n * mu) <= 420 * (c**2) * logT:
        a = np.random.randint(0, k)
        # r = np.random.random() < means[a] 
        r = sample_reward(means[a], sigma2, test_type)

        n[a] += 1
        sums[a] += r
        mu[a] = sums[a] / n[a]
        arms[t] = a
        t += 1

    # print(t)
    # Phase 2: NCB selection
    while t < T:
        ncb_values = np.zeros(k)

        # bonus = np.empty(k, dtype=np.float64)
        for i in range(k):
            if n[i] > 0:
                
                bonus = 2 * c * np.sqrt((2 * mu[i] * logT) / n[i])
                ncb_values[i] = mu[i] + bonus
            else:
                ncb_values[i] = 1e9

        # ncb = mu + bonus
        A = np.argmax(ncb_values)
        r = sample_reward(means[A], sigma2, test_type)


        n[A] += 1
        sums[A] += r
        mu[A] = sums[A] / n[A]
        arms[t] = A
        t += 1

    return arms

def simulate_ncb(means, T_max, sigma2,  num_trials, c, test_type, regret_type):
    """
    Returns an array of length T_max where entry t-1 is the average Nash regret at time t,
    averaged over num_trials independent runs.
    """
    mu_star = np.max(means)
    total_rewards = []

    for _ in tqdm(range(num_trials), desc=f"NCB Non-Private Trials"):
        arms = NCB_single(np.array(means), sigma2, c, T_max, test_type)
        rewards = np.array(means)[arms]                # length T_max
        rewards.reshape(1, T_max)
        total_rewards.append(rewards)

    total_rewards = np.array(total_rewards)
    expected_means = np.sum(total_rewards, axis = 0)/num_trials
    if regret_type == "Nash":

        cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-300)))   # shape (T_max,)
        inv_t = 1.0 / np.arange(1, T_max+1)
        geom_mean = np.exp(cumsum_log * inv_t)           # shape (T_max,)
        avg_regret = mu_star - geom_mean                  # shape (T_max,)
        return avg_regret
    else:
        cum_rewards = np.cumsum(expected_means)         # shape (T_max,)
        inv_t     = 1.0 / np.arange(1, T_max+1)         # 1/t for t=1..T_max

        arith_mean = cum_rewards * inv_t                # shape (T_max,)

        avg_regret = mu_star - arith_mean               # shape (T_max,)

        return avg_regret

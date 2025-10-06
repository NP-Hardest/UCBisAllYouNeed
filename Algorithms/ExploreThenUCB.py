import numpy as np
from numba import njit
from tqdm import tqdm


@njit
def clip(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x
    

@njit    
def sample_reward(mean, sigma2, test_type):
    """
    Sample a reward according to the arm's mean category,
    implemented in pure nopython with only random.random().
    """

    u = np.random.rand()  
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
def ExploreUCB_single(means, T, sigma2, test_type, p):
    """
    Implementation of the UCB algorithm from the image.
    Parameters: Time horizon T, number of arms k, exploration period tilde_T.
    """
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu_hat = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)
    
    # Set exploration period (tilde_T) - typically sqrt(T) or similar
    den = np.log(k) if p >= 0 else 1
    if (p==0):
        tilde_T =  int(16 * np.sqrt(T * k * np.log(T)/den))
    elif(p>0): 
        tilde_T =  int(16 * np.sqrt(T * np.pow(float(k), p) * np.log(T)/den))
    else:
        tilde_T =  16 * np.sqrt(T  * np.log(T)/np.pow(float(k), -p))
    # print(np.pow(float(k),p))
    t = 0
    # ---- Phase 1: Exploration ----
    while t < tilde_T:
        # Uniformly sample arm
        i_t = np.random.randint(k)
        r = sample_reward(means[i_t], sigma2, test_type)
        
        n[i_t] += 1
        sums[i_t] += r
        mu_hat[i_t] = sums[i_t] / n[i_t]
        arms[t] = i_t
        t += 1
    
    
    # print(t)
    # ---- Phase 2: Exploitation with UCB ----
    while t <= T:
        ucb_values = np.zeros(k)
        
        for i in range(k):
            if n[i] > 0:
                # UCB formula from the image: μ̂_i + 4 * sqrt(log(T)/n_i)
                bonus = 4 * np.sqrt(np.log(T) / n[i])
                ucb_values[i] = mu_hat[i] + bonus
            else:
                # If arm hasn't been pulled, give it maximum priority
                ucb_values[i] = 1e9
        
        # Select arm with highest UCB
        i_t = np.argmax(ucb_values)
        r = sample_reward(means[i_t], sigma2, test_type)
        
        n[i_t] += 1
        sums[i_t] += r
        mu_hat[i_t] = sums[i_t] / n[i_t]
        arms[t] = i_t
        t += 1
    
    return arms

def simulate_ExploreUCB_single(means, T_max, sigma2, num_trials, test_type, p=0):
    mu_star = np.max(means)
    total_rewards = []

    for _ in tqdm(range(num_trials), desc=f"Explore-then-UCB Trials (p={p})"):
        arms = ExploreUCB_single(means, T_max, sigma2, test_type, p)
        rewards = np.array(means)[arms]
        total_rewards.append(rewards)

    total_rewards = np.array(total_rewards)
    expected_means = np.sum(total_rewards, axis=0) / num_trials

    if p == 0:  
        cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-300)))
        inv_t = 1.0 / np.arange(1, T_max+1)
        geom_mean = np.exp(cumsum_log * inv_t)
        avg_regret = mu_star - geom_mean
    elif p == 1:  
        cum_rewards = np.cumsum(expected_means)
        inv_t = 1.0 / np.arange(1, T_max+1)
        arith_mean = cum_rewards * inv_t
        avg_regret = mu_star - arith_mean
    else:  
        p_powers = np.power(expected_means, p)               # x^p (safe because of clipping)
        # print(p_powers)
        cum_p_powers = np.cumsum(p_powers)
        inv_t = 1.0 / np.arange(1, T_max + 1)
        p_mean = np.power(cum_p_powers * inv_t, 1.0 / p)
        avg_regret = mu_star - p_mean
        # return avg_regret

        
        
    return avg_regret
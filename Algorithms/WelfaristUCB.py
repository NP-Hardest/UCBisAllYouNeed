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

           
        #     x = u ** 0.25
        #     return 1.0 if x > 1.0 else (0.0 if x < 0.0 else x)

        # elif mean >= 0.25 and mean < 0.5:
          
        #     return 0.4 if u < 0.5 else 1.0

        # elif mean < 0.25:
        
        #     return u

        # else:
       
        #     return 1.0 if u < mean else 0.0

# @njit
# def arm_thresholds(mu, n, sigma2, logT, c):
#     sqrt_term = np.zeros_like(mu)
#     safe_n = np.maximum(n, 1)
#     sqrt_term[n>0] = np.sqrt((2*c*sigma2*logT)/safe_n[n>0])
#     diff = mu - sqrt_term
#     # if np.any(diff <= 0):
#     #     return np.full(len(mu), 1e9, dtype=np.float64) 
#     # if diff<=0, we force a tiny positive number so denom->∞ and paper‐term->∞
#     diff = np.maximum(diff, 1e-20)

#     paper_term = (c**2 * sigma2 * logT) / diff
#     return 400 * (paper_term) \
#             + c*np.sqrt(2 * n * sigma2 * logT)     




@njit
def phase1_condition(mu, n, sigma2, logT, c, p):
    k = len(mu)
    for i in range(k):
        if n[i] == 0:  # unplayed arm -> condition holds
            continue

        bonus = c * np.sqrt((2 * sigma2 * logT) / n[i])
        lhs1 = mu[i] <= bonus

        denom = mu[i] - bonus
        if denom <= 0:
            rhs_term = 1e18  # huge so condition holds
        else:
            rhs_term = (200 * (c**2) * (p**2) * sigma2 * logT) / denom \
                       + c * np.sqrt(2 * n[i] * sigma2 * logT)

        lhs2 = (n[i] * mu[i]) < rhs_term

        # If both fail -> stop Phase I
        if not (lhs1 or lhs2):
            return False
    return True

# @njit
@njit
def Welfarist_UCB_single(means, T, sigma2, test_type, p):
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)
    
    c = 2
    t = 0
    logT = np.log(T)
    # ---- Phase 1 ----
    B = np.arange(k)
    np.random.shuffle(B)
    b_index = 0

    p_a = 1 if p>=-1 else p

    while phase1_condition(mu, n, sigma2, logT, c, p_a):
        if t % k == 0:
            B = np.arange(k)
            np.random.shuffle(B)
            b_index = 0

        a = B[b_index]
        b_index += 1

        r = sample_reward(means[a], sigma2, test_type)
        n[a] += 1
        sums[a] += r
        mu[a] = sums[a] / n[a]
        arms[t] = a
        t += 1
        if t >= T:
            break

    # print("Phase I ended at t =", t)

    # ---- Phase 2 ----
    while t < T:
        ucb = np.zeros(k)
        for a in range(k):
            if n[a] > 0:
                bonus = c * np.sqrt((2 * sigma2 * logT) / n[a])
                ucb[a] = mu[a] + bonus
            else:
                ucb[a] = 1e9  # force exploration
        A = np.argmax(ucb)
        r = sample_reward(means[A], sigma2, test_type)
        n[A] += 1
        sums[A] += r
        mu[A] = sums[A] / n[A]
        arms[t] = A
        t += 1
    
    return arms


def simulate_Welfarist_UCB_single(means, T_max, sigma2, num_trials, test_type, p):
    mu_star = np.max(means)
    total_rewards = []

    for _ in tqdm(range(num_trials), desc=f"Welfarist UCB Trials Trials (p={p})"):
        arms = Welfarist_UCB_single(means, T_max, sigma2, test_type, p)
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
        return avg_regret

        
        
    return avg_regret


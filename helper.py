import matplotlib.pyplot as plt
import numpy as np



def plot_expt_a(regrets, T_max, flag):
    ts = np.arange(1, T_max+1)

    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    

        
    mask = ((ts >= 1e5) & (ts<= T_max) if not flag else  (ts >= 1e2) & (ts<= T_max) )
    # mask = (ts >= 1e2) & (ts<= T_max)      

    plt.figure(figsize=(8, 5))
    
     
    nash_regret_standard_ncb = regrets[0][idxs]
    nash_regret_welfarist_ucb = regrets[1][idxs]


    expt = "a" if not flag else "b"

    plt.plot(ts[mask], nash_regret_standard_ncb[mask], label=rf"Standard NCB", color="orange")
    plt.plot(ts[mask], nash_regret_welfarist_ucb[mask], label=rf"Welfarist UCB")

    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel("Nash Regret", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 15})
    plt.xscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_{expt}.png")
    plt.show()


def plot_expt_b(regrets, T_max, p):
    ts = np.arange(1, T_max+1)

    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    

        
    mask = ((ts >= 1e2) & (ts<= T_max) )
    # mask = (ts >= 1e2) & (ts<= T_max)      

    plt.figure(figsize=(8, 5))
    
     
    p_regret_explore_ucb = regrets[0][idxs]
    p_regret_welfarist_ucb = regrets[1][idxs]


    if(p==0.5):
        expt = "c"
    elif (p==-0.5):
        expt = "d"
    elif (p==-1.5):
        expt = "e"

    plt.plot(ts[mask], p_regret_explore_ucb[mask], label=rf"Explore-then-UCB, $p={p}$", color="green")
    plt.plot(ts[mask], p_regret_welfarist_ucb[mask], label=rf"Welfarist UCB, $p={p}$")

    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel("Nash Regret", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 15})
    plt.xscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_{expt}.png")
    plt.show()



def plot_expt_f(regrets, T_max):
    ts = np.arange(1, T_max+1)


    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    
    p_regret_ncb_1 = regrets[0][idxs]
    p_regret_ncb_0 = regrets[1][idxs]
    p_regret_ncb_m1 = regrets[2][idxs]
    p_regret_ncb_m2 = regrets[3][idxs]
    p_regret_ncb_m5  = regrets[4][idxs]

        
    mask = (ts >= 1e2) & (ts<= T_max)       # to plot between 1e6 and 6e6


    plt.figure(figsize=(8, 5))


    plt.plot(ts[mask], p_regret_ncb_1[mask], label=rf"Welfarist UCB, $p=1$")
    plt.plot(ts[mask], p_regret_ncb_0[mask], label=rf"Welfarist UCB, $p=0$")
    plt.plot(ts[mask], p_regret_ncb_m1[mask], label=rf"Welfarist UCB, $p=-1$" )
    plt.plot(ts[mask], p_regret_ncb_m2[mask], label=rf"Welfarist UCB, $p=-2$")
    plt.plot(ts[mask], p_regret_ncb_m5[mask], label=rf"Welfarist UCB, $p=-5$")


    plt.xscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel(rf"$p-$mean Regret", fontsize=16)
    plt.legend(loc="upper right", prop={'size': 15})
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_f.png")
    plt.show()



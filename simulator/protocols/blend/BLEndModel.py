# implementations of the equations used by the blend model
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def sigma(s, L, b):
    return 0.5 * (s/(L - b))

def gamma(C, s, L, b):
    return (C - 2) * (1 + sigma(s, L, b))

# number of beacons potentially colliding within window W
def omega(C, s, L, b, nb):
    return 1 + (W(s, nb)/(L - b)) * (gamma(C, s, L, b) - 1)

# average size of the window of contention across all beacons in the epoch
def W(s, nb):
    return (s * nb)/4   # using definition from optimizer which differs from paper

# probability of no collision
def Pnc(C, s, L, b):
    return math.pow((1 - ((2 * b)/(L - b))), gamma(C, s, L, b))

# probability of not having a collision in an epoch k > 1
def PncW(C, s, L, b, nb):
    if W(s, nb) < (L - b):
        return math.pow((1 - ((2 * b)/W(s, nb))), omega(C, s, L, b, nb))
    else:
        return Pnc(C, s, L, b)

def Pd(Na, tau, Omega):
    E = Omega.iloc[0]['E']      # epoch length

    C = Na                      # node density
    s = Omega.iloc[0]['s']      # max random slack
    L = Omega.iloc[0]['L']      # listening interval
    b = Omega.iloc[0]['b']      # beacon length
    nb = Omega.iloc[0]['nb']    # number of beacons

    Lambda = tau        # tau is a window size, lambda is a discovery latency

    k = math.floor(Lambda/E)    # number of epochs

    return 1 - (1 - Pnc(C, s, L, b)) * math.pow((1 - PncW(C, s, L, b, nb)), k - 1) * (1 - (((Lambda - (k * E))/E)) * PncW(C, s, L, b, nb))

# NOTE beacon length should be input as 1 even if b = 3 in the schedule bc the optimizer computes the schedule with b/3
# same as Pd but accepts individual args rather than dataframe
def compute_Pd(Ne, Lambda, E, s, L, b, nb):
    k = math.floor(Lambda/E)    # number of epochs
    return 1 - (1 - Pnc(Ne, s, L, b)) * math.pow((1 - PncW(Ne, s, L, b, nb)), k - 1) * (1 - (((Lambda - (k * E))/E)) * PncW(Ne, s, L, b, nb))

# run on command line
if __name__ == '__main__':
    # read in the computed schedules
    df = pd.read_csv('./schedules_backup.csv')

    # set a desired set of req for the schedule we're evaluating
    P = 0.9
    Lambda = 2000
    Ne = 50

    # pick that schedule from the dataframe
    Omega = df.loc[(df['N'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]

    # create a list of actual node densities and tau window sizes to test
    Nas = list(range(10, 60, 10))
    taus = list(range(1000, 60100, 10))

    fig, ax = plt.subplots()

    all_Pds = []

    # make a plot for each actual node density
    for Na in Nas:
        Pds = [round(Pd(Na, x, Omega), 3) for x in taus]
        ax.plot(taus, Pds, label=f'{Na}')
        all_Pds.append(Pds)
        print(f'Na = {Na}')
        print(f'\ttau for 100% discovery = {taus[Pds.index(1.0)]}')
        print(f'\ttau for 90% discovery = {taus[Pds.index(0.90)]}')

        
    plt.xlabel('tau')
    plt.ylabel('$P_d$')
    plt.suptitle(f'$P_d$ of schedule as window size increases (varying $N_a$)')
    plt.title(f'$N_e = {Omega.iloc[0]["N"]}$, $P = {Omega.iloc[0]["P"]}$, $\Lambda = {Omega.iloc[0]["Lambda"]}$')
    plt.legend()
    plt.show()

    target_Pds = np.linspace(0.9, 1.0, num=10)

    fig, ax = plt.subplots()

    # make a plot showing the tau required to attain a given Pd for each Na
    for i, Na in enumerate(Nas):
        ys = []

        for Pd in target_Pds:
            ys.append(taus[all_Pds[i].index(round(Pd, 3))])

        ax.plot(target_Pds, ys, label=f'{Na}')

    plt.xlabel('$P_d$')
    plt.ylabel('tau')
    plt.suptitle(f'tau required for target $P_d$ (varying $N_a$)')
    plt.title(f'$N_e = {Omega.iloc[0]["N"]}$, $P = {Omega.iloc[0]["P"]}$, $\Lambda = {Omega.iloc[0]["Lambda"]}$')
    plt.legend()
    plt.show()
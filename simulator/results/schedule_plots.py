""" Contains functions for plotting:
    - Different schedules optimized for different requirements
    - Comparisons between energy consumption for Ne and varying Na
"""
from math import floor
from os import popen
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# battery capacity
BATTERY = (225 * 3 / 2.1)

# global dataframe containing schedules that have already been computed (so we can avoid recomputing them)
df = pd.read_csv('../protocols/blend/schedules.csv')

def optimize(P: float, Lambda: int, N: int, beacon_duration, max_random_slack, is_ublend: bool):
    """ Optimize a set of BLEnd parameters to the given requirements.
    
    This function invokes the BLEnd optimizer to derive the appropriate protocol
    parameters for the given constraints.

    Args:
        P: minimum floating point discovery probability
        Lambda: maximum integer discovery latency (in ms)
        N: expected integer number of nodes in collision domain
        beacon_duration: b duration of beacon in ms
        max_random_slack: max slack between beacons in ms
        is_ublend: boolean mode value that determines whether schedule is for ublend or fblend
    """
    global df

    # check if the schedule has been pre-computed and saved in the table already
    schedule = df.loc[(df['P'] == P) & (df['Lambda'] == Lambda) & (df['N'] == N) & (df['b'] == beacon_duration) & (df['s'] == max_random_slack)]

    if schedule.empty:
        # run the optimizer and store the json console output in params
        stream = popen(f'Rscript ../blend/optimizer.R {P} {Lambda} {N} {beacon_duration} {max_random_slack} {int(not is_ublend)}')
        params = json.loads(stream.read())
        
        schedule = pd.DataFrame({
            'is_ublend': is_ublend,
            'P': P,
            'Lambda': Lambda,
            'N': N,
            'b': beacon_duration,
            's': max_random_slack,
            'E': params['E'],
            'A': params['A'],
            'L': params['A'] + beacon_duration + max_random_slack,
            'nb': params['nb'],
            'Q': params['Q(E,A)']
        }, index=[0])

        df = df.append(schedule)

    return schedule
    
def chart_schedules(P: float, Lambda: int, Ns: list, beacon_duration, max_random_slack, is_ublend: bool):
    """ Plots a schedule's E, A, and L (for a given P and Lambda requirement) 
    with N on the x-axis and duration (in ms) on the y-axis
    """
    Es = []
    As = []
    Ls = []

    for N in Ns:
        schedule = optimize(P, Lambda, N, beacon_duration, max_random_slack, is_ublend)

        Es.append(schedule['E'])
        As.append(schedule['A'])
        Ls.append(schedule['L'])

    plt.plot(Ns, Es, '*-', label='epoch')
    plt.plot(Ns, As, 's-', label='advertising interval')
    plt.plot(Ns, Ls, 'o-', label='scan interval')

    plt.xlabel('N')
    plt.ylabel('ms')

    plt.title(f'BLEnd schedules for varying N; P = {P}, Lambda = {Lambda}ms')

    plt.legend()

    plt.savefig(f'./figures/schedules/P={P},Lam={Lambda},b={beacon_duration},slack={max_random_slack},ublend={is_ublend}.pdf')
    
    plt.show()
    plt.clf()

def chart_energy_comparison(s1, s2, duration):
    """ Charts a comparison of energy consumption of two schedules over the given duration.
    Schedules should be an indexable dict or dataframe, duration should be in hours.
    """
    # create x axis values in ms starting from 0 to duration in 1 minute increments
    xs = range(0, duration * 3600000, 60000)
    s1ys = [s1['Q'] * floor(x/s1['E']) for x in xs]
    s2ys = [s2['Q'] * floor(x/s2['E']) for x in xs]

    plt.plot(xs, s1ys, 'b', label=f'N = {s1.iloc[0]["N"]}')
    plt.plot(xs, s2ys, 'r', label=f'N = {s2.iloc[0]["N"]}')
    
    plt.legend()

    plt.show()

def bar_chart_energy_comparison_diff_estimates(s_es, s_a, duration):
    """ Bar chart of estimated N power consumption vs actual N power consumption over the given duration in hrs.
    Charts s_es (estimated N schedules) vs s_a (actual/correct N schedule). """

    labels = [f'N = {s.iloc[0]["N"]}' for s in s_es]
    

    print(labels)
    actuals = [(s_a.iloc[0]['Q']/1000) * duration * 3600000 for _ in range(0, len(s_es))]
    estimates = [(s.iloc[0]['Q']/1000) * duration * 3600000 for s in s_es]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, actuals, width, label='Actual N = 2')
    rects2 = ax.bar(x + width/2, estimates, width, label='Estimated N', color='tomato')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('power cost (Ah)')
    ax.set_title(f'Relative power cost of different estimated schedules over {duration} hours')
    
    # hacky way to fix labels
    labels.insert(0, labels[0])

    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

def bar_chart_energy_comparison_diff_actuals(s_as, s_e):
    """ Bar chart of actual N power consumption vs an estimated N power consumption.
    Charts s_as (actual N schedules) vs s_e (estimated N schedule). """

    labels = ['$N_{a}$ = ' + f'{s.iloc[0]["N"]}' for s in s_as]
    
    actuals = [s.iloc[0]['Q'] for s in s_as]
    estimates = [s_e.iloc[0]['Q'] for _ in range(0, len(s_as))]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, actuals, width, label='$N_{a}$')
    rects2 = ax.bar(x + width/2, estimates, width, label='$N_{e}$ = ' + f'{s_e.iloc[0]["N"]}', color='tomato')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('instantaneous current (mAh)')
    ax.set_title('Power cost of schedule for $N_{e} = ' + f'{s_e.iloc[0]["N"]}' + '$ vs. schedules for $N_{a}$')
    
    # hacky way to fix labels
    labels.insert(0, labels[0])

    ax.set_xticklabels(labels)

    plt.savefig('./figures/Ne_vs_Na.pdf')
    plt.show()

def bar_chart_lifetime_comparison(s_as, s_e):
    """ Bar chart of actual N lifetime vs an estimated N lifetime.
    Charts s_as (actual N schedules) vs s_e (estimated N schedule). 
    """

    labels = ['$N_{a}$ = ' + f'{s.iloc[0]["N"]}' for s in s_as]
    
    actuals = [(BATTERY/s.iloc[0]['Q'])/24 for s in s_as]
    estimates = [(BATTERY/s_e.iloc[0]['Q'])/24 for _ in range(0, len(s_as))]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, actuals, width, label='$N_{a}$')
    rects2 = ax.bar(x + width/2, estimates, width, label='$N_{e}$ = ' + f'{s_e.iloc[0]["N"]}', color='tomato')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('lifetime (days)')
    ax.set_title('Battery lifetime of schedule for $N_{e} = ' + f'{s_e.iloc[0]["N"]}' + '$ vs. schedules for $N_{a}$')
    
    # hacky way to fix labels
    labels.insert(0, labels[0])

    ax.set_xticklabels(labels)

    plt.savefig('./figures/Ne_vs_Na.pdf')
    plt.show()

def compute_schedules(Ps, Lambdas, Ns):
    # generate a bunch of schedules - at what point does the optimizer break down for different input requirements?
    for P in Ps:
        for Lambda in Lambdas:
            for N in Ns:
                 schedule = optimize(P, Lambda, N, 3, 10, False)
                 df.to_csv('../blend/schedules.csv', index=False)


def chart_many_schedules(Ps, Lambdas, Ns):
    for P in Ps:
        for Lambda in Lambdas:
            chart_schedules(P, Lambda, Ns, 3, 10, False)

P = 0.9
Lambda = 6000

N_as = [2, 10, 20, 30, 40, 50]
N_e = 50

s_as = [df.loc[(df['P'] == P) & (df['Lambda'] == Lambda) & (df['N'] == N)] for N in N_as]
s_e = df.loc[(df['P'] == P) & (df['Lambda'] == Lambda) & (df['N'] == N_e)]

bar_chart_energy_comparison_diff_estimates(s_as, s_e)
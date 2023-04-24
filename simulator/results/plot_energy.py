import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import math
import pickle

from statistics import mean, stdev
from scipy import integrate
import numpy as np

import results.colors as colors

# hack the path so i can import the BLEnd optimizer...
import sys, os
sys.path.append(os.path.abspath('../protocols'))

from blend.OptimizerDriver import optimize

BATTERY = (225 * 3 / 2.1)
SCHEDULES = pd.read_csv('../protocols/blend/schedules.csv')

def compute_lifetimes(currents):
    lifetimes = [BATTERY/current/24 for current in currents]
    return lifetimes

def compute_currents(df, Q_key='Q'):
    """ Compute the overall current consumption using the discrete integral.

    This uses the given Q_key to grab the column containing the current samples we want to integrate.
    """
    return [integrate.simpson(df.loc[df['node'] == node][Q_key])/len(df.loc[df['node'] == node]) for node in pd.unique(df['node'])]

def compute_optimal_blend_currents(P, Lambda, df, Na_key='Na'):
    """ Computes the optimal current usage across the duration of the sim in the given df for BLEnd.

    This is determined by invoking the optimizer for each of the different P, Lambda, and Na_key 
    values observed at each epoch of the simulation.
    """
    for Na in pd.unique(df[Na_key]):
        try:
            df.loc[df[Na_key] == Na, 'optimal_Q'] = get_blend_Q(P, Lambda, math.ceil(Na / 5) * 5) # round Na to the nearest 5 since that's what the nodes are doing
        except:
            print(f'Optimal schedule config for Na = {Na} not found in the table. Invoking optimizer...')
            schedule = optimize(P, Lambda, Na, 3, 10, False)
            df.loc[df[Na_key] == Na, 'optimal_Q'] = schedule['Q']

    return compute_currents(df, Q_key='optimal_Q')

def compute_PRR_schedule(Ne):
    """ Computes the probabilities of each slot in the Birthday schedule for Ne nodes """
    pt = 1/Ne
    pl = 1 - (1/Ne)
    ps = 0
    return pt, pl, ps

def get_blend_Q(P, Lambda, N):
    """ Returns the cost of a BLEnd schedule optimized to P, Lambda, and N nodes """
    return SCHEDULES.loc[(SCHEDULES['P'] == P) & (SCHEDULES['Lambda'] == Lambda) & (SCHEDULES['N'] == N)]['Q'].item()

def get_bday_Q(N):
    """ Returns the cost of a Birthday schedule optimized to N nodes """
    I_SCAN = 6.329
    I_ADV = 5.725
    I_IDLE = 0.08064

    pt, pl, ps = compute_PRR_schedule(N)
    return (pt * I_ADV) + (pl * I_SCAN) + (ps * I_IDLE)

def compute_bday_currents(df, N_key='Na', Q_key='optimal_Q'):
    """ Computes the overall cost of the Bday schedule used by each node over the duration of the sim in the dataframe.
    
    Depending on the N_key given, this can either compute the cost of the schedule the node actually used (if N_key = Ne), 
    or it can be the cost of the OPTIMAL schedule (if N_key = Na)
    """
    df[Q_key] = df[N_key].apply(lambda N : get_bday_Q(N))
    return compute_currents(df, Q_key=Q_key)

def get_currents(s_path, a_path, protocol, P=0.9, Lambda=8000):
    print(f'getting currents for {s_path} and {a_path}, protocol = {protocol}')

    cols = ['node', 'Q', 'Ne', 'Na']

    # process the sim output to get the plot data for blend
    s_df = pd.read_csv(s_path, usecols=cols)
    print(f'read s_df = {s_path}')

    a_df = pd.read_csv(a_path, usecols=cols)
    print(f'read a_df = {a_path}')

    if protocol == 'blend':
        print(f'computing s_currents...')
        s_currents = compute_currents(s_df)
    
        print(f'computing a_currents...')
        a_currents = compute_currents(a_df)

        print(f'computing e_currents...')
        e_currents = compute_optimal_blend_currents(P, Lambda, s_df)

    elif protocol == 'bday':
        print(f'computing s_currents...')
        s_currents = compute_bday_currents(s_df, N_key='Ne', Q_key='schedule_Q')

        print(f'computing a_currents...')
        a_currents = compute_bday_currents(a_df, N_key='Ne', Q_key='schedule_Q')

        print(f'computing e_currents...')
        e_currents = compute_bday_currents(s_df, N_key='Na')

    print(f'done computing currents.')
    return s_currents, a_currents, e_currents

def get_chartable_lifetimes(s_currents, a_currents, e_currents):
    print('computing lifetimes...')

    lifetimes = [compute_lifetimes(c) for c in [s_currents, a_currents, e_currents]]
    lt_min = min(flat(lifetimes))
    lt_max = max(flat(lifetimes))
    lifetimes = [[normalize(x, lt_min, lt_max) for x in lt] for lt in lifetimes]

    print('done computing lifetimes.')

    static_lifetime = mean(lifetimes[0])

    # the average delta between the static baseline and the adaptive and exact lifetimes
    means = {
        'adaptive': mean(lifetimes[1]) - static_lifetime,
        'exact': mean(lifetimes[2]) - static_lifetime
    }

    errors = {
        'adaptive': confidence_interval(lifetimes[1]),
        'exact': confidence_interval(lifetimes[2])
    }

    return means, errors

def get_chartable_lifetimes_dict(currents):
    return get_chartable_lifetimes(currents['static'], currents['adaptive'], currents['exact'])

def normalize(x, xmin, xmax):
    return (x - xmin)/(xmax - xmin)

def confidence_interval(samples, z=1.960):
    return (z * (stdev(samples)/math.sqrt(len(samples))))

def flat(l):
    return [item for sublist in l for item in sublist]

def bar_chart_lifetime_comparison_static_baseline(scenario, fig_path=None, pickle_path=None):
    """ Charts the energy usage of adaptive and exact schedules as a normalized delta from 
    the lifetime of the static schedule as a baseline.
    """
    labels = [trial[0] for trial in scenario['data']]

    blend_currents, bday_currents = get_currents_from_scenario_dict(scenario)

    blend_means, blend_errors = get_chartable_lifetimes(blend_currents['static'], blend_currents['adaptive'], blend_currents['exact'])
    bday_means, bday_errors = get_chartable_lifetimes(bday_currents['static'], bday_currents['adaptive'], bday_currents['exact'])

    adaptive_lifetimes = [
        blend_means['adaptive'],
        bday_means['adaptive']
    ]

    adaptive_errs = [
        blend_errors['adaptive'],
        bday_errors['adaptive']
    ]

    exact_lifetimes = [
        blend_means['exact'],
        bday_means['exact']
    ]

    exact_errs = [
        blend_errors['exact'],
        bday_errors['exact']
    ]

    x = np.arange(len(labels))  # the label locations
    width = 0.1                 # the width of the bars
    capsize = 5                 # size of caps for error bars

    fig, ax = plt.subplots()
    rects2 = ax.bar(x - width/2, adaptive_lifetimes, width, yerr=adaptive_errs, capsize=capsize, label='adaptive', color=colors.SOFT_ADAPTIVE_COLOR)
    rects3 = ax.bar(x + width/2, exact_lifetimes, width, yerr=exact_errs, capsize=capsize, label='exact', color=colors.SOFT_OPTIMAL_COLOR)

    ax.set_xticks(x, labels)

    plt.legend()

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

def get_currents_from_scenario_dict(ds):
    blend_currents = get_currents_for_trial(ds['data'][0], 'blend')
    bday_currents = get_currents_for_trial(ds['data'][1], 'bday')

    return blend_currents, bday_currents

def get_currents_for_trial(trial, protocol):
    pickle_path = trial[3]
    print(f'currents from path {pickle_path}')

    cur = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pickle_path = os.path.join(cur, 'results', pickle_path)

    # pickle already exists, load it and return the results
    if os.path.exists(pickle_path):
        print(f'loading existing pickle')
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            currents = data

    # otherwise process the results and save a pickle
    else:
        print(f'computing new currents (could take a while...)')
        if protocol == 'blend':
            s_currents, \
            a_currents, \
            e_currents = get_currents(trial[1], trial[2], protocol, P=trial[4], Lambda=trial[5])
        
        elif protocol == 'bday':
            s_currents, \
            a_currents, \
            e_currents = get_currents(trial[1], trial[2], protocol)

        currents = {
            'static': s_currents,
            'adaptive': a_currents,
            'exact': e_currents
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(currents, f)

    return currents

def bar_chart_all_lifetime_comparison_static_baseline(ds1, ds2, ds3, fig_path=None):    
    labels = [trial[0] for trial in ds1['data']] + [trial[0] for trial in ds2['data']] + [trial[0] for trial in ds3['data']]

    # compute currents for each scenario
    bl_inc_currents, bd_inc_currents = get_currents_from_scenario_dict(ds1)
    bl_dec_currents, bd_dec_currents = get_currents_from_scenario_dict(ds2)
    bl_lev_currents, bd_lev_currents = get_currents_from_scenario_dict(ds3)

    # blend and bday density increase lifetimes and errors
    bl_inc_means, bl_inc_errors = get_chartable_lifetimes_dict(bl_inc_currents)
    bd_inc_means, bd_inc_errors = get_chartable_lifetimes_dict(bd_inc_currents)

    # blend and bday density decrease lifetimes and errors
    bl_dec_means, bl_dec_errors = get_chartable_lifetimes_dict(bl_dec_currents)
    bd_dec_means, bd_dec_errors = get_chartable_lifetimes_dict(bd_dec_currents)

    # blend and bday levy walk lifetimes and errors
    bl_lev_means, bl_lev_errors = get_chartable_lifetimes_dict(bl_lev_currents)
    bd_lev_means, bd_lev_errors = get_chartable_lifetimes_dict(bd_lev_currents)

    means = [bl_inc_means, bd_inc_means, bl_dec_means, bd_dec_means, bl_lev_means, bd_lev_means]
    errors = [bl_inc_errors, bd_inc_errors, bl_dec_errors, bd_dec_errors, bl_lev_errors, bd_lev_errors]

    adaptive_lifetimes = [ls['adaptive'] for ls in means]
    adaptive_errors = [ls['adaptive'] for ls in errors]
    exact_lifetimes = [ls['exact'] for ls in means]
    exact_errors = [ls['exact'] for ls in errors]

    x = np.arange(len(labels))  # the label locations
    width = 0.2                 # the width of the bars
    capsize = 5                 # size of caps for error bars

    fig, ax = plt.subplots()
    ax.set_ylim([-1.0, 1.0])

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(1.5, color='black', linewidth=0.5)
    ax.axvline(3.5, color='black', linewidth=0.5)

    rects2 = ax.bar(x - width/2, adaptive_lifetimes, width, yerr=adaptive_errors, capsize=capsize, label='adaptive', color=colors.SOFT_ADAPTIVE_COLOR)
    rects3 = ax.bar(x + width/2, exact_lifetimes, width, yerr=exact_errors, capsize=capsize, label='exact', color=colors.SOFT_OPTIMAL_COLOR)

    ax.set_xticks(x, labels)
    ax.set_ylabel('difference in lifetime')

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    fig.text(0.55, 1.04, ds1['name'], va='center', ha='center', transform=trans)
    fig.text(2.55, 1.04, ds2['name'], va='center', ha='center', transform=trans)
    fig.text(4.5, 1.04, ds3['name'], va='center', ha='center', transform=trans)

    plt.legend(loc='upper right')

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

def get_performance_difference(P, path):
    try:
        df = pd.read_csv(path, usecols=['node', 'discovery_rate'])
        df = df[df['node'].isna()]
    except ValueError:
        df = pd.read_csv(path, usecols=['discovery_rate'])

    cumulative = integrate.simpson(df['discovery_rate'])/len(df)
    P_int = integrate.simpson(np.full(len(df), P))/len(df)

    return P_int, cumulative

def energy_vs_benefit(scenario, fig_path=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    plot_energy_vs_benefit(scenario, ax1, ax2)

    plt.legend()

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

def condensed_energy_vs_benefit(s1, s2, fig_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    
    condensed_plot_energy_vs_benefit(s1, s2, axs[0], axs[1])

    fig.text(0.215, 0.982, s1['name'], va='center', ha='center')
    fig.text(0.4, 0.982, s2['name'], va='center', ha='center')

    fig.text(0.7, 0.982, s1['name'], va='center', ha='center')
    fig.text(0.883, 0.982, s2['name'], va='center', ha='center')

    axs[0].set_ylabel('lifetime')
    axs[1].set_ylabel('performance')

    fig.tight_layout()
    plt.legend(loc='lower right', prop={'size': 9.5})

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.0)

    plt.show()

def condensed_plot_energy_vs_benefit(s1, s2, ax1, ax2):
    # plot the left chart (battery lifetime)
    labels = [trial[0] for trial in s1['data']] + [trial[0] for trial in s2['data']]

    # compute currents for each scenario
    blc1, bdc1 = get_currents_from_scenario_dict(s1)
    blc2, bdc2 = get_currents_from_scenario_dict(s2)

    # blend and bday density increase lifetimes and errors
    bl_inc_means1, bl_inc_errors1 = get_chartable_lifetimes_dict(blc1)
    bl_inc_means2, bl_inc_errors2 = get_chartable_lifetimes_dict(blc2)
    
    bd_inc_means1, bd_inc_errors1 = get_chartable_lifetimes_dict(bdc1)
    bd_inc_means2, bd_inc_errors2 = get_chartable_lifetimes_dict(bdc2)

    means1 = [bl_inc_means1, bd_inc_means1]
    errors1 = [bl_inc_errors1, bd_inc_errors1]
    means2 = [bl_inc_means2, bd_inc_means2]
    errors2 = [bl_inc_errors2, bd_inc_errors2]

    adaptive_lifetimes = [ls['adaptive'] for ls in means1] + [ls['adaptive'] for ls in means2]
    adaptive_errors = [ls['adaptive'] for ls in errors1] + [ls['adaptive'] for ls in errors2]
    exact_lifetimes = [ls['exact'] for ls in means1] + [ls['exact'] for ls in means2]
    exact_errors = [ls['exact'] for ls in errors1] + [ls['exact'] for ls in errors2]

    x = np.arange(len(labels), dtype=float)  # the label locations

    width = 0.15                # the width of the bars
    capsize = 5                 # size of caps for error bars

    ax1.set_ylim([-1.0, 1.0])    
    ax1.set_xlim([-0.5, 3.5])

    ax1.bar(x - width/2, adaptive_lifetimes, width, yerr=adaptive_errors, capsize=capsize, label='adaptive', color=colors.SOFT_ADAPTIVE_COLOR)
    ax1.bar(x + width/2, exact_lifetimes, width, yerr=exact_errors, capsize=capsize, label='exact', color=colors.SOFT_OPTIMAL_COLOR)

    ax1.set_xticks(x, labels)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(1.5, color='black', linewidth=0.5)

    P = s1['data'][0][4]
    P2 = s2['data'][0][4]

    # plot the right: difference in performance
    pbls, bls = get_performance_difference(P, s1['performance'][0][1])
    pbla, bla = get_performance_difference(P, s1['performance'][0][2])

    pbds, bds = get_performance_difference(P, s1['performance'][1][1])
    pbda, bda = get_performance_difference(P, s1['performance'][1][2])

    pbls2, bls2 = get_performance_difference(P2, s2['performance'][0][1])
    pbla2, bla2 = get_performance_difference(P2, s2['performance'][0][2])

    pbds2, bds2 = get_performance_difference(P2, s2['performance'][1][1])
    pbda2, bda2 = get_performance_difference(P2, s2['performance'][1][2])

    ax2.set_ylim([-1.0, 1.0])
    ax2.set_xlim([-0.5, 3.5])

    ax2.bar(x - width/2, [bla - bls, bda - bds] + [bla2 - bls2, bda2 - bds2], width, label='adaptive', color=colors.SOFT_ADAPTIVE_COLOR)
    ax2.bar(x + width/2, [pbls - bls, pbds - bds] + [pbls2 - bls2, pbds2 - bds2], width, label='optimal', color=colors.SOFT_OPTIMAL_COLOR)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(1.5, color='black', linewidth=0.5)

    ax2.set_xticks(x, labels)

def four_energy_vs_benefit(s1, s2, fig_path=None):
    fig, axs = plt.subplots(2, 2)

    plot_energy_vs_benefit(s1, axs[0, 0], axs[1, 0])
    plot_energy_vs_benefit(s2, axs[0, 1], axs[1, 1])

    fig.text(0.315, 0.982, s1['name'], va='center', ha='center')
    fig.text(0.79, 0.982, s2['name'], va='center', ha='center')

    axs[0, 0].set_ylabel('lifetime')
    axs[1, 0].set_ylabel('performance')

    fig.tight_layout()
    plt.legend(loc='lower right')

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

def plot_energy_vs_benefit(scenario, ax1, ax2):
    # plot the left chart (battery lifetime)
    labels = [trial[0] for trial in scenario['data']]

    # compute currents for each scenario
    bl_inc_currents, bd_inc_currents = get_currents_from_scenario_dict(scenario)

    # blend and bday density increase lifetimes and errors
    bl_inc_means, bl_inc_errors = get_chartable_lifetimes_dict(bl_inc_currents)
    bd_inc_means, bd_inc_errors = get_chartable_lifetimes_dict(bd_inc_currents)

    means = [bl_inc_means, bd_inc_means]
    errors = [bl_inc_errors, bd_inc_errors]

    adaptive_lifetimes = [ls['adaptive'] for ls in means]
    adaptive_errors = [ls['adaptive'] for ls in errors]
    exact_lifetimes = [ls['exact'] for ls in means]
    exact_errors = [ls['exact'] for ls in errors]

    x = np.arange(len(labels), dtype=float)  # the label locations

    width = 0.15                # the width of the bars
    capsize = 5                 # size of caps for error bars

    ax1.set_ylim([-1.0, 1.0])
    ax2.set_ylim([-1.0, 1.0])
    
    ax1.set_xlim([-0.5, 1.5])
    ax2.set_xlim([-0.5, 1.5])

    ax1.bar(x - width/2, adaptive_lifetimes, width, yerr=adaptive_errors, capsize=capsize, label='adaptive', color=colors.SOFT_ADAPTIVE_COLOR)
    ax1.bar(x + width/2, exact_lifetimes, width, yerr=exact_errors, capsize=capsize, label='exact', color=colors.SOFT_OPTIMAL_COLOR)

    ax1.set_xticks(x, labels)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)

    P = scenario['data'][0][4]

    # plot the right: difference in performance
    pbls, bls = get_performance_difference(P, scenario['performance'][0][1])
    pbla, bla = get_performance_difference(P, scenario['performance'][0][2])

    pbds, bds = get_performance_difference(P, scenario['performance'][1][1])
    pbda, bda = get_performance_difference(P, scenario['performance'][1][2])

    ax2.bar(x - width/2, [bla - bls, bda - bds], width, label='adaptive', color=colors.SOFT_ADAPTIVE_COLOR)
    ax2.bar(x + width/2, [pbls - bls, pbds - bds], width, label='optimal', color=colors.SOFT_OPTIMAL_COLOR)

    ax2.set_xticks(x, labels)
    
##
## final figures
##
def estimated_vs_actual_energy_figure():
    bar_chart_lifetime_comparison(0.9, 10000, 50, [2, 10, 20, 30, 40, 50], fig_path='./figures/energy_difference.pdf')

def dynamic_increase_energy_figure(scenario):
    bar_chart_lifetime_comparison_static_baseline(
        scenario=scenario,
        fig_path='./figures/dynamic_increase_energy.pdf'
    )

def dynamic_decrease_energy_figure(scenario):
    bar_chart_lifetime_comparison_static_baseline(
        scenario=scenario,
        fig_path='./figures/dynamic_decrease_energy.pdf'
    )

def all_energy_figure(s1, s2, s3):
    bar_chart_all_lifetime_comparison_static_baseline(
        ds1=s1,
        ds2=s2,
        ds3=s3,
        fig_path='./figures/all_energy.pdf'
    )

dd_increase_scenario = {
    'name': 'increase',
    'data': [
        ('BLEnd', './data/blend_static_energy_dd.csv', './data/blend_adaptive_energy_dd.csv', './data/blend_di.pkl', 0.9, 8000),
        ('Bday', './data/bday_static_energy_dd.csv', './data/bday_adaptive_energy_dd.csv', './data/bday_di.pkl'),
    ],
    'performance': [
        ('BLEnd', './data/blend_static_dd.csv', './data/blend_adaptive_dd.csv', 0.9, 8000),
        ('Bday', './data/bday_static_dd.csv', './data/bday_adaptive_dd.csv'),
    ]
}

dd_decrease_scenario = {
    'name': 'decrease',
    'data': [
        ('BLEnd', './data/blend_static_energy_dd_decrease.csv', './data/blend_adaptive_energy_dd_decrease.csv', './data/blend_dd.pkl', 0.9, 8000),
        ('Bday', './data/bday_s_dd_energy.csv', './data/bday_a_dd_energy.csv', './data/bday_dd2.pkl')
    ],
    'performance': [
        ('BLEnd', './data/blend_static_dd_decrease.csv', './data/blend_adaptive_dd_decrease.csv', 0.9, 8000),
        ('Bday', './data/bday_s_dd.csv', './data/bday_a_dd.csv')
    ]
}

levy_walk_low_scenario = {
    'name': '$N_e = 20$',
    'data': [
        ('BLEnd', './data/blend_slw_ne20_energy.csv', './data/blend_alw_ne20_energy.csv', './data/blend_lw_ne20.pkl', 0.9, 10000),
        ('Bday', './data/bday_static_energy_levy_walk.csv', './data/bday_adaptive_energy_levy_walk.csv', './data/bday_lw_ne20.pkl')
    ],
    'performance': [
        ('BLEnd', './data/0_blend_slw_ne20.csv', './data/0_blend_alw_ne20.csv', 0.9, 10000),
        ('Bday', './data/0_bday_slw_ne20.csv', './data/0_bday_alw_ne20.csv')
    ]
}

levy_walk_high_scenario = {
    'name': '$N_e = 60$',
    'data': [
        ('BLEnd', './data/blend_slw_ne60_energy.csv', './data/blend_alw_ne60_energy.csv', './data/blend_lw_ne60.pkl', 0.9, 10000),
        ('Bday', './data/bday_slw_ne60_energy.csv', './data/bday_alw_ne60_energy.csv', './data/bday_lw_ne60.pkl')
    ],
    'performance': [
        ('BLEnd', './data/blend_slw_ne60.csv', './data/blend_alw_ne60.csv', 0.9, 10000),
        ('Bday', './data/bday_slw_ne60.csv', './data/bday_alw_ne60.csv')
    ]
}

condensed_energy_vs_benefit(dd_increase_scenario, dd_decrease_scenario, fig_path='./figures/condensed_energy_v_cost.pdf')
condensed_energy_vs_benefit(levy_walk_low_scenario, levy_walk_high_scenario, fig_path='./figures/condensed_mobility_energy_v_cost.pdf')
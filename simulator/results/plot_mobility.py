import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import results.colors as colors

def plot_trace(df):
    fig, ax = plt.subplots()

    ax.set_ylim([-100, 100])
    ax.set_xlim([-100, 100])

    plt.plot(df['pos_x'], df['pos_y'])
    plt.show()

def plot_num_neighbors(df):
    fig, ax = plt.subplots()

    plt.xlabel('time (ms)')
    plt.ylabel('$N_a$')

    plt.plot(df['t'], df['average_Na'])
    plt.show()

def plot_discovery_rate(df):
    fig, ax = plt.subplots()

    ax.set_ylim([0, 1.0])

    plt.xlabel('time (ms)')
    plt.ylabel('discovery rate')

    plt.plot(df['t'], df['discovery_rate'])
    plt.show()

def plot_stacked(P, Ne, fns, units='ms', xlim=None, outfile=None, Ne_color='orange', Na_color='gray'):
    """ Generates a figure with two subplots, the top showing discovery rate vs time
    and the bottom showing actual number of neighbors vs time """
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_ylim([0.0, 1.1])
    ax2.set_ylim([-15, 95])

    if xlim is not None:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

    lss = [':', '-']

    if units == 'min':
        scale = 60000
    elif units == 'hrs':
        scale = 3600000
    else:
        scale = 1

    for i, f in enumerate(fns):
        df = pd.read_csv(f[0])
        df = df[~df['node'].isna()]

        start_step = df['t'].iloc[0]
        df['t'] = df['t'].apply(lambda x : (x - start_step)/scale)

        ax1.plot(df['t'], df['discovery_rate'].rolling(scale).mean(), label=f'{f[1]} {f[2]}', ls=lss[i], color=f[3])

        if f[4]:
            # figure out where adaptation time changes so we can plot vertical lines
            df['adaptation_change'] = df['average_Ne'].diff()
            adaptations = df[df['adaptation_change'] != 0]

    # target discovery probability
    ax1.axhline(P, ls='--', color=colors.INDICATOR_LINE_COLOR)

    ax1.legend(loc='lower right')

    # plot adaptations
    for a in adaptations['t'].unique():
        if a > 0:
            # nan out the edges where Ne changes so that the line doesnt have vertical jumps in it
            df.loc[df['t'] == a, 'average_Ne'] = np.nan
            #ax2.axvline(a, ls='--', color=colors.INDICATOR_LINE_COLOR)
    
    # static Ne estimate
    # ax2.axhline(Ne, ls='--', color=colors.SOFT_STATIC_COLOR)

    ax2.plot(df['t'], df['average_Na'], label='$N_a$', color=colors.NEUTRAL_COLOR)
    ax2.plot(df['t'], df['average_Ne'], label='$N_e\'$', color=Ne_color)
    ax2.legend(loc='upper right')

    plt.xlabel(f'time ({units})')

    ax1.set_ylabel('discovery rate')
    ax2.set_ylabel('neighbors')

    # fig.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile)

    plt.show()

def levy_walk_bday_ne30_ego_figure():
    plot_stacked(
        0.9,
        30,
        [
            ('./data/0_bday_static_levy_walk.csv', 'static', 'Bday, $N_e = 30$', colors.STATIC_COLOR, False), 
            ('./data/0_bday_adaptive_levy_walk.csv', 'adaptive', 'Bday', colors.ADAPTIVE_COLOR, True)
        ],
        units='min',
        xlim=[-1, 31],
        Ne_color=colors.ADAPTIVE_COLOR,
        outfile='./figures/bday_ego_mobility.pdf'
    )

def levy_walk_blend_ne2_ego_figure():
    plot_stacked(
        0.9,
        2,
        [
            ('./data/0_blend_static_levy_walk.csv', 'static', 'BLEnd, $N_e = 2$', colors.STATIC_COLOR, False), 
            ('./data/0_blend_adaptive_levy_walk.csv', 'adaptive', 'BLEnd', colors.ADAPTIVE_COLOR, True)
        ],
        units='min',
        xlim=[-1, 31],
        Ne_color=colors.ADAPTIVE_COLOR,
        outfile='./figures/blend_ego_mobility.pdf'
    )

levy_walk_blend_ne2_ego_figure()
levy_walk_bday_ne30_ego_figure()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import math

import results.colors

def plot_cdf(P, Lambda, fn, title=''):
    df = pd.read_csv(f'./{fn}')

    start_step = df['t'].iloc[0]

    df['t'] = df['t'].apply(lambda x : x - start_step)

    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.1])

    ax.plot(df['t'], df['discovery_rate'])

    plt.axhline(P, ls='--', color=colors.INDICATOR_LINE_COLOR)
    plt.axvline(Lambda, ls='--', color=colors.INDICATOR_LINE_COLOR)

    plt.title(title)
    plt.ylabel('discovery rate')
    plt.xlabel('time (ms)')

    plt.show()

def plot_cdfs(P, fns, title='', outfile=None):
    fig, ax = plt.subplots()

    # ax.set_ylim([0.0, 1.0])

    lss = [':', '-']

    for i, fn in enumerate(fns):
        df = pd.read_csv(f'./{fn[0]}')
        df = df[df['node'].isna()]

        start_step = df['t'].iloc[0]

        df['t'] = df['t'].apply(lambda x : x - start_step)

        # figure out where Na changes so we can plot vertical lines
        df['Na_change'] = df['Na'].diff()
        na_changes = df[df['Na_change'] != 0]

        # rolling mean makes bday prettier but is not necessary for blend
        ax.plot(df['t'], df['discovery_rate'].rolling(1000).mean(), label=f'{fn[1]} {fn[2]}', ls=lss[i])

    plt.axhline(P, ls='--', color=colors.INDICATOR_LINE_COLOR)
    
    # plot node density changes
    for change in na_changes['t'].unique():
        if change > 0:
            plt.axvline(change, ls='--', color=colors.INDICATOR_LINE_COLOR)

    plt.legend()
    plt.title(title)
    plt.ylabel('discovery rate')

    # bday is slots, blend is time in ms 
    if fn[0][2] == 'Bday':
        plt.xlabel('slots')
    else:
        plt.xlabel('time (ms)')

    if outfile is not None:
        plt.savefig(outfile)
        
    plt.show()

def plot_stacked(P, fns1, fns2, outfile=None, title='', units='ms', xlim=None):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_ylim([0.0, 1.1])
    ax2.set_ylim([0.0, 1.1])

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

    for i, fn in enumerate(fns1):
        df = pd.read_csv(f'./{fn[0]}')

        if 'node' in df:
            df = df[df['node'].isna()]

        start_step = df['t'].iloc[0]

        df['t'] = df['t'].apply(lambda x : (x - start_step)/scale)

        # figure out where Na changes so we can plot vertical lines
        df['Na_change'] = df['Na'].diff()
        na_changes = df[df['Na_change'] != 0]

        # birthday discovery rates appear really noisy since they vary each slot (rather than in a time window),
        # so smooth it out a little bit with a rolling mean
        ax1.plot(df['t'], df['discovery_rate'].rolling(fn[4]).mean(), label=f'{fn[1]} {fn[2]}', ls=lss[i%len(lss)], color=fn[3])
    
    # create a blended transform to convert data coordinates on the plot to drawing coordinates
    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    first = True

    # plot node density changes
    for change in na_changes['t'].unique():
        if change > 0:
            ax1.axvline(change, ls='--', color=colors.INDICATOR_LINE_COLOR)
            
            if first:
                ax1.text(change - (55000/scale), 1.065, f'$N_a$ = {int(df.loc[df["t"] == change]["Na"])}', transform=trans, va='center')
                first = False
            else:
                ax1.text(change - (8000/scale), 1.07, f'{int(df.loc[df["t"] == change]["Na"])}', transform=trans, va='center')

    for i, fn in enumerate(fns2):
        df = pd.read_csv(f'./{fn[0]}')

        if 'node' in df:
            df = df[df['node'].isna()]
        
        start_step = df['t'].iloc[0]

        df['t'] = df['t'].apply(lambda x : (x - start_step)/scale)

        # figure out where Na changes so we can plot vertical lines
        df['Na_change'] = df['Na'].diff()
        na_changes = df[df['Na_change'] != 0]

        # rolling mean makes bday prettier but is not necessary for blend
        ax2.plot(df['t'], df['discovery_rate'].rolling(fn[4]).mean(), label=f'{fn[1]} {fn[2]}', ls=lss[i%len(lss)], color=fn[3])

    # plot node density changes
    for change in na_changes['t'].unique():
        if change > 0:
            ax2.axvline(change, ls='--', color=colors.INDICATOR_LINE_COLOR)

    ax1.axhline(P, ls='--', color=colors.INDICATOR_LINE_COLOR)
    ax2.axhline(P, ls='--', color=colors.INDICATOR_LINE_COLOR)

    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')

    plt.xlabel(f'time ({units})')

    #fig.text(0.04, 0.5, 'discovery rate', va='center', rotation='vertical')

    # tighter layout
    # fig.text(0.015, 0.5, 'discovery rate', va='center', rotation='vertical')
    fig.supylabel('discovery rate', fontsize=10)
    # fig.tight_layout()
    # plt.subplots_adjust(left = 0.1)

    if outfile is not None:
        plt.savefig(outfile)

    plt.show()

##
## final charts
##
def simple_performance_comparison_figure():
    plot_stacked(
        0.9, 
        [
            ('./data/simple_bday.csv', 'static', 'Bday, $N_e = 30$', colors.STATIC_COLOR, 2000), 
            ('./data/simple_bday_adaptive.csv', 'adaptive', 'Bday', colors.ADAPTIVE_COLOR, 2000)
        ],
        [
            ('./data/simple_blend.csv', 'static', 'BLEnd, $N_e = 2$', colors.STATIC_COLOR, 1000), 
            ('./data/simple_blend_adaptive.csv', 'adaptive', 'BLEnd', colors.ADAPTIVE_COLOR, 1000),
        ],
        outfile='./figures/simple_comparison.pdf',
        units='min',
        xlim=[0, 3]
    )

def density_increase_figure():
    # dynamic node density, stacked chart
    plot_stacked(
        0.9, 
        [
            ('./data/bday_static_dd.csv', 'static', 'Bday', colors.STATIC_COLOR, 6000), 
            ('./data/bday_adaptive_dd.csv', 'adaptive', 'Bday', colors.ADAPTIVE_COLOR, 6000)
        ],
        [
            ('./data/blend_static_dd_7k.csv', 'static', 'BLEnd', colors.STATIC_COLOR, 6000), 
            ('./data/blend_adaptive_dd_7k.csv', 'adaptive', 'BLEnd', colors.ADAPTIVE_COLOR, 6000),
        ],
        outfile='./figures/density_increase.pdf',
        units='min',
        xlim=[0, 10]
    )

def density_decrease_figure():
    # dynamic node density, stacked chart
    plot_stacked(
        0.9, 
        [
            ('./data/bday_s_dd.csv', 'static', 'Bday', colors.STATIC_COLOR, 6000), 
            ('./data/bday_a_dd.csv', 'adaptive', 'Bday', colors.ADAPTIVE_COLOR, 6000)
        ],
        [
            ('./data/blend_static_dd_decrease.csv', 'static', 'BLEnd', colors.STATIC_COLOR, 6000), 
            ('./data/blend_adaptive_dd_decrease.csv', 'adaptive', 'BLEnd', colors.ADAPTIVE_COLOR, 6000),
        ],
        outfile='./figures/density_decrease.pdf',
        units='min',
        xlim=[0, 10]
    )

def levy_walk_figure():
    # levy walk mobility, stacked chart
    plot_stacked(
        0.9, 
        [
            ('./data/0_bday_slw_ne20.csv', 'static', 'Bday', colors.STATIC_COLOR, 6000), 
            ('./data/0_bday_alw_ne20.csv', 'adaptive', 'Bday', colors.ADAPTIVE_COLOR, 6000)
        ],
        [
            ('./data/0_blend_slw_ne20.csv', 'static', 'BLEnd', colors.STATIC_COLOR, 6000), 
            ('./data/0_blend_alw_ne20.csv', 'adaptive', 'BLEnd', colors.ADAPTIVE_COLOR, 6000)
        ],
        outfile='./figures/levy_walk_ne20.pdf',
        units='min',
        xlim=[-1, 31]
    )

# simple_performance_comparison_figure()
# density_increase_figure()
# density_decrease_figure()
# levy_walk_figure()
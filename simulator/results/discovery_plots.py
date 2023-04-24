# plots the data from the discovery rate and latency csv files generated by the evaluators
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

import results.colors

def plot_discovery_rate(P, Lambda, Ne, output_filename='./discovery_rate.csv'):
    df = pd.read_csv(output_filename)

    df = df.loc[(df['Ne'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]

    xs = pd.unique(df['Na'])
    y1s = []

    # take the average over all runs performed for each Na
    for Na in xs:
        y1s.append(df.loc[(df['Na'] == Na)]['discovery_rate'].mean()*100)
    
    y2s = np.array([P*100 for _ in range(0, len(y1s))])

    fig, ax = plt.subplots()

    ax.plot(xs, y1s, marker='.')
    ax.plot(xs, y2s, '--', color='dimgray')

    # ax.fill_between(xs, y1s, y2s, where=(y1s > y2s), color='C0', alpha=0.3, interpolate=True)
    ax.fill_between(xs, y1s, y2s, where=(y1s <= y2s), color='C1', alpha=0.2, interpolate=True)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.ylabel('discovery rate')
    plt.xlabel('$N_a$')
    # plt.title(f'Decline in discovery rate of schedule for $N_e = {Ne}$ as $N_a$ grows')

    plt.xticks(np.arange(10, max(xs)+1, 10))

    plt.savefig(f'./figures/discovery_rate_Ne={Ne};P={P};Lambda={Lambda}.pdf')
    plt.show()

def plot_discovery_latency(P, Lambda, Ne, output_filename='./discovery_latency.csv'):
    df = pd.read_csv(output_filename)

    df = df.loc[(df['Ne'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]

    xs = pd.unique(df['Na'])
    y1s = []

    # take the average over all runs performed for each Na
    for Na in xs:
        y1s.append(df.loc[(df['Na'] == Na)]['discovery_latency'].mean())

    y2s = np.array([Lambda for _ in range(0, len(y1s))])
    
    fig, ax = plt.subplots()

    ax.plot(xs, y1s, marker='.')
    ax.plot(xs, y2s, '--', color='dimgray')

    # ax.fill_between(xs, y1s, y2s, where=(y1s <= y2s), color='C0', alpha=0.3, interpolate=True)
    ax.fill_between(xs, y1s, y2s, where=(y1s > y2s), color='C1', alpha=0.2, interpolate=True)

    plt.ylabel('discovery latency (ms)')
    plt.xlabel('$N_a$')
    # plt.title(f'Increase in discovery latency of schedule for $N_e = {Ne}$ as $N_a$ grows')

    plt.xticks(np.arange(10, max(xs)+1, 10))

    plt.savefig(f'./figures/discovery_latency_Ne={Ne};P={P};Lambda={Lambda}.pdf')
    plt.show()

def plot_discovery_latencies(P, Ne):
    df = pd.read_csv('./discovery_latency.csv')

    df = df.loc[(df['Ne'] == Ne) & (df['P'] == P)]

    xs = pd.unique(df['Na'])

    fig, ax = plt.subplots()

    for Lambda in pd.unique(df['Lambda']):
        ys = df.loc[df['Lambda'] == Lambda]['discovery_latency']

        ax.plot(xs, ys, marker='.', label=f'$\Lambda = {Lambda}$')
        plt.axhline(Lambda, ls='--', color='dimgray')

    plt.legend()
    plt.ylabel('discovery latency (ms)')
    plt.xlabel('$N_a$')
    plt.title(f'Increase in discovery latency of schedules for $N_e = {Ne}$ as $N_a$ grows')

    plt.xticks(np.arange(10, max(xs)+1, 10))

    plt.savefig(f'./figures/discovery_latencies_Ne={Ne};P={P};Lambda={Lambda}.pdf')
    plt.show()

def plot_discovery_rate_cdfs(P, Lambda, Ne, output_filename='continuous_discovery_rate.csv'):
    df = pd.read_csv(f'./{output_filename}')

    # grab the subset of the dataset with the target estimate value
    df = df.loc[(df['Ne'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]

    fig, ax = plt.subplots()

    Nas = list(pd.unique(df['Na']))
    Nas.sort()

    for Na in Nas:
        xs = df.loc[df['Na'] == Na]['time_elapsed']
        ys = df.loc[df['Na'] == Na]['discovery_rate']

        if Na == Ne:
            ls = ':'
        else:
            ls = '-'

        ax.plot(xs, ys, ls=ls, label=f'{Na}')

        # print the "optimal tau", i.e. the point where P = 1.0 for this schedule (window size to discover all neighbors)
        print(f'Na = {Na}')
        #print(f'\tOptimal Tau = {df.loc[(df["Na"] == Na) & (df["discovery_rate"] >= 1.0)]["time_elapsed"].iloc[0]}')

    plt.axhline(P, ls='--', color='dimgray')
    plt.axvline(Lambda, ls='--', color='dimgray')

    plt.legend()
    plt.ylabel('average discovery rate')
    plt.xlabel('time (ms)')
    plt.title(f'$N_e = {Ne}$, $P = {P}$, $\Lambda = {Lambda}$')
    plt.suptitle(f'CDFs of varying $N_a$')

    plt.savefig(f'./figures/continuous_discovery_rate_Ne={Ne};P={P};Lambda={Lambda}.pdf')
    plt.show()

def plot_Nd_histogram(P, Lambda, Ne, output_filename='./incremental_window_distribution.csv'):
    df = pd.read_csv(output_filename)

    df = df.loc[(df['Ne'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]

    trials = pd.unique(df['Na'])

    fig, ax = plt.subplots()

    ignore = []

    for Na in trials:
        if Na not in ignore:
            total_samples = df.loc[df['Na'] == Na]['frequency'].sum()
            ys = df.loc[df['Na'] == Na]['frequency'].apply(lambda x: x/total_samples)

            if Na == Ne:
                ls = ':'
            else:
                ls = '-'

            ax.plot(df.loc[df['Na'] == Na]['Ndiff'], ys, ls=ls, label=f'$N_a$ = {Na}')
 
    plt.legend()
    
    plt.xlabel('$N_{diff_a}$')
    plt.ylabel('frequency')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    
    plt.suptitle(f'Distribution of $N_{{diff_a}}$ for $N_e = {Ne}$, varying $N_a$')
    plt.title(f'$P = {P}$, $\Lambda = {Lambda}$')
    
    #plt.savefig(f'./figures/{output_filename}.pdf')
    plt.show()

def plot_stacked(P, Lambda, Ne, dr_path, dl_path):
    # plot_discovery_rate(P, Lambda, Ne, dr_path)
    # plot_discovery_latency(P, Lambda, Ne, dl_path)
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ##
    ## plot discovery rate up top
    ##
    df = pd.read_csv(dr_path)
    df = df.loc[(df['Ne'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]

    xs = pd.unique(df['Na'])
    y1s = []

    # take the average over all runs performed for each Na
    for Na in xs:
        y1s.append(df.loc[(df['Na'] == Na)]['discovery_rate'].mean())
    
    y2s = np.array([P for _ in range(0, len(y1s))])

    ax1.plot(xs, y1s, marker='.')
    ax1.plot(xs, y2s, '--', color=colors.INDICATOR_LINE_COLOR)
    ax1.fill_between(xs, y1s, y2s, where=(y1s <= y2s), color=colors.SOFT_STATIC_COLOR, alpha=0.2, interpolate=True)
    #ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_ylabel('discovery rate')

    ##
    ## plot discovery latency on the bottom
    ##
    df = pd.read_csv(dl_path)
    df = df.loc[(df['Ne'] == Ne) & (df['Lambda'] == Lambda) & (df['P'] == P)]
    df['discovery_latency'] = df['discovery_latency'].apply(lambda x : x/1000)

    xs = pd.unique(df['Na'])
    y1s = []

    # take the average over all runs performed for each Na
    for Na in xs:
        y1s.append(df.loc[(df['Na'] == Na)]['discovery_latency'].mean())

    y2s = np.array([Lambda/1000 for _ in range(0, len(y1s))])

    ax2.plot(xs, y1s, marker='.')
    ax2.plot(xs, y2s, '--', color=colors.INDICATOR_LINE_COLOR)
    ax2.fill_between(xs, y1s, y2s, where=(y1s > y2s), color=colors.SOFT_STATIC_COLOR, alpha=0.2, interpolate=True)

    ax2.set_ylabel('discovery latency (sec)')
    plt.xlabel('$N_a$')

    plt.savefig('./figures/performance_loss.pdf')
    plt.show()

plot_stacked(0.9, 4000, 20, './data/discovery_rate_degradation.csv', './data/discovery_latency_degradation.csv')

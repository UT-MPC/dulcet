
from os import popen, path
import json
import logging
import csv

import pandas as pd

cur = path.dirname(path.abspath(__file__))
df_fn = path.join(cur, 'schedules.csv')

header = ['is_ublend', 'P', 'P_expected', 'Lambda' , 'N', 'b', 's', 'E', 'A', 'L', 'nb', 'Q']

if not path.exists(df_fn):
    with open(df_fn, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

df = pd.read_csv(df_fn)

def configuration_dict(schedule):
    return {
        'is_ublend': schedule.iloc[0]['is_ublend'],
        'P': schedule.iloc[0]['P'],
        'P_expected': schedule.iloc[0]['P_expected'],
        'Lambda': schedule.iloc[0]['Lambda'],
        'N': schedule.iloc[0]['N'],
        'b': schedule.iloc[0]['b'],
        's': schedule.iloc[0]['s'],
        'E': schedule.iloc[0]['E'],
        'A': schedule.iloc[0]['A'],
        'L': schedule.iloc[0]['L'],
        'nb': schedule.iloc[0]['nb'],
        'Q': schedule.iloc[0]['Q']
    }

def optimize(P: float, Lambda: int, N: int, b: int, s: int, is_ublend: bool):
    """ Optimize a set of BLEnd parameters to the given requirements.
    
    This function invokes the BLEnd optimizer in ./blend/ to derive the appropriate protocol
    parameters for the given constraints.

    Args:
        P: minimum floating point discovery probability
        Lambda: maximum integer discovery latency (in ms)
        N: expected integer number of nodes in collision domain
    """
    global df

    schedule = df.loc[(df['P'] == P) & (df['Lambda'] == Lambda) & (df['N'] == N) & (df['b'] == b) & (df['s'] == s)]

    # only compute the schedule if we haven't precomputed it already
    if schedule.empty:
        cur = path.dirname(path.abspath(__file__))
        opt_fn = path.join(cur, 'optimizer.R')
        
        logging.info(f'[OptimizerDriver] Invoking schedule optimizer for P = {P}, Lambda = {Lambda}, N = {N}, unidirectional = {is_ublend}/{int(is_ublend)}')
        stream = popen(f'Rscript {opt_fn} {P} {Lambda} {N} {b} {s} {int(is_ublend)}')
        params = json.loads(stream.read())

        schedule = pd.DataFrame({
            'is_ublend': is_ublend,
            'P': P,
            'P_expected': params['P'],
            'Lambda': Lambda,
            'N': N,
            'b': b,
            's': s,
            'E': params['E'],
            'A': params['A'],
            'L': params['A'] + b + s,
            'nb': params['nb'],
            'Q': params['Q(E,A)']
        }, index=[0])

        df = df.append(schedule)

        logging.info(f'[OptimizerDriver] Saving new schedule for P = {P}, Lambda = {Lambda}, N = {N}')
        df.to_csv(df_fn, index=False)

    return configuration_dict(schedule)
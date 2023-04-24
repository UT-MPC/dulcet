# playing around with different schedules and computations of Ndiffe
from os import path, popen
import json
import csv

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd 

import protocols.blend.OptimizerDriver
# import OptimizerDriver

def run_Rscript(Rscript, args):
    cur = path.dirname(path.abspath(__file__))
    script_fn = path.join(cur, Rscript)

    args = [str(x) for x in args]
    arg_string = ' '.join(args)

    stream = popen(f'Rscript {script_fn} {arg_string}')
    output = json.loads(stream.read())

    return output

def PdLambda(N, Lambda, schedule):
    return run_Rscript('model.R', [Lambda, N, schedule['E'], schedule['A'], schedule['nb'], 1])['P']

def Ndiff(N, schedule):
    schedule['beacon_duration'] = 3

    PdL = PdLambda(N, schedule['Lambda'], schedule)
    Pd2L = PdLambda(N, 2*schedule['Lambda'], schedule)

    # return (Pd2L * (N - 1)) - (PdL * (N - 1))
    return (Pd2L * N) - (PdL * N)

def optimize(P, Lambda, Ne):
    return OptimizerDriver.optimize(P, Lambda, Ne, 3, 10, False)

def Ndiffe_table(P, Lambda, Nes):
    print(list(Nes))
    for Ne in Nes:
        schedule = optimize(P, Lambda, Ne)
        print(schedule)
        print(f'Ndiffe = {Ndiffe(schedule["N"], schedule)}')

# run some Ndiff computations (and save the output) to validate sim results
def simulation_validation(P, Lambda):
    Nas = range(10, 110, 10)
    Nes = range(10, 60, 10)

    print(Nas)
    print(Nes)

    for Ne in Nes:
        schedule = optimize(P, Lambda, Ne)
        print(schedule)

        for Na in Nas:

            # an i for each window
            for i in [0, 1]:
                header = ['P', 'P_schedule_best', 'Lambda', 'Ne', 'Na', 'window', 'Pdw', 'computed_Ndiff']

                cur = path.dirname(path.dirname(path.abspath(__file__)))
                fn = path.join(cur, f'results/Ndiff_P={P}-Lambda={Lambda}.csv')

                if not path.exists(fn):
                    with open(fn, mode='w') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                with open(fn, mode='a') as f:
                    row = [
                        P,
                        schedule['P_expected'],
                        schedule['Lambda'],
                        Ne,
                        Na,
                        schedule['Lambda'] * (i + 1),
                        PdLambda(Na, Lambda * (i + 1), schedule),
                        Ndiffe(Na, schedule) * i    # hacky, so Ndiff is 0 for first window
                    ]

                    writer = csv.writer(f)
                    writer.writerow(row)

# generate data for a 3d plot of Na vs Ne vs Ndiff for a given schedule
def Ndiff_3d(P, Lambda):
    Nas = range(10, 55, 5)
    Nes = range(10, 55, 5)
    Ndiffs = []

    for Ne in Nes:
        schedule = optimize(P, Lambda, Ne)
        print(schedule)

        for Na in Nas:
            header = ['Ne', 'Na', 'computed_Ndiff']

            cur = path.dirname(path.dirname(path.abspath(__file__)))
            fn = path.join(cur, f'results/Ndiff_3d_P={P}-Lambda={Lambda}.csv')

            if not path.exists(fn):
                with open(fn, mode='w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

            with open(fn, mode='a') as f:
                row = [
                    Ne,
                    Na,
                    Ndiffe(Na, schedule)
                ]

                writer = csv.writer(f)
                writer.writerow(row)

def plot_Ndiff_3d(fn):
    df = pd.read_csv(fn)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(df['Na'], df['Ne'], df['computed_Ndiff'])

    # ax.plot_surface(df['Na'], df['Ne'], df['computed_Ndiff'], cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)

    ax.set_xlabel('$N_a$')
    ax.set_ylabel('$N_e$')
    ax.set_zlabel('$N_{diff}$')

    ax.azim = -135
    ax.elev = 15

    #plt.title('Expected difference in neighbors discovered between windows\nSchedules for varying $N_e$, $P=0.9$, $\Lambda=4000$')
    #plt.title('Schedules for varying $N_e$, $P=0.9$, $\Lambda=4000$')
    plt.tight_layout()

    plt.savefig('./schedule_topography.pdf')
    plt.show()

# Ndiff_3d(0.9, 4000)
# plot_Ndiff_3d('./Ndiff_3d_P=0.9-Lambda=4000.csv')
import pandas as pd
import matplotlib.pyplot as plt

def plot_normalized_comparison(fns):
    xs = []
    ys = []

    for fn in fns:
        df = pd.read_csv(fn)
        df['avg_epsilon'] = df['avg_epsilon'] / df['avg_epsilon'].abs().max()
        # df['avg_epsilon'] = (df['avg_epsilon'] - df['avg_epsilon'].mean()) / df['avg_epsilon'].std()

        xs.append(df['t'])
        ys.append(df['avg_epsilon'])

    lses = ['-', ':', '--']

    for i, x in enumerate(xs):
        plt.plot(x, ys[i], ls=lses[i])

    plt.ylabel('$\epsilon$')
    plt.xlabel('t')

    #plt.savefig('./figures/bday_eps-Ne30Na20.pdf')
    plt.show()

def compile_dataframe(fn, series_length=1000):
    df = pd.read_csv(fn)

    data = {
        'Na': [],
        'Ne': [], 
        'Nep_mean': [], 
        'Nep_stddev': [], 
        'eps_mean': [], 
        'eps_stddev': []
    }

    for Na in df['Na'].unique():
        for Ne in df.loc[df['Na'] == Na]['Ne'].unique():
            tmp = df.loc[(df['Na'] == Na) & (df['Ne'] == Ne)].tail(series_length)

            data['Na'].append(Na)
            data['Ne'].append(Ne)
            data['Nep_mean'].append(tmp['Nep'].mean())
            data['Nep_stddev'].append(tmp['Nep'].std())
            data['eps_mean'].append(tmp['epsilon'].mean())
            data['eps_stddev'].append(tmp['epsilon'].std())
        
    ndf = pd.DataFrame().from_dict(data)
    
    return ndf

def plot_eps_subplot(ax, ndf):
    lss = ['-', ':', '--', '-.']

    for i, Ne in enumerate(ndf['Ne'].unique()):
        ax.plot(ndf.loc[ndf['Ne'] == Ne]['Na'], ndf.loc[ndf['Ne'] == Ne]['Nep_mean'], ls=lss[i%(len(lss))], label=f'$N_e = {Ne}$')

def plot_stacked(bday, blend, fig_path=None):
    bl_df = compile_dataframe(blend)
    bd_df = compile_dataframe(bday, series_length=100)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    plot_eps_subplot(ax1, bd_df)
    plot_eps_subplot(ax2, bl_df)

    plt.xlabel('$N_a$')
    ax1.set_ylabel('$N_e\'$')
    ax2.set_ylabel('$N_e\'$')
   
    ax3 = ax1.twinx()
    ax3.get_yaxis().set_ticks([])
    ax3.set_ylabel('Bday')

    ax4 = ax2.twinx()
    ax4.get_yaxis().set_ticks([])
    ax4.set_ylabel('BLEnd')

    ax1.legend()

    fig.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

plot_stacked('./data/bday_adaptive_all.csv', './data/blend_adaptive_all.csv', './figures/sensing_performance.pdf')
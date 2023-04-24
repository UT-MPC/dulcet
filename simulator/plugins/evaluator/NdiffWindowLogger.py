import csv
import statistics
from os import path, remove
import logging
import random
import sys
import pandas as pd

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

""" Evaluates additional nodes discovered for different multiples of the window size:

t ----------------------------->
    [  Lambda  ][  Lambda  ]
    |   Nd1    |
    |   Nd2                |
  
t ----------------------------->
    [  Lambda  ][  Lambda  ][  Lambda  ]
    |   Nd1    |
    |   Nd2                            |  

...

t ----------------------------->
    [  Lambda  ][  Lambda  ]    ...    [  Lambda  ]
    |   Nd1    |
    |   Nd2                                       |  

    Ndiff = Nd2 - Nd1
"""
class NdiffWindowLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the window size (usually lambda)
        self.window = kwargs['window']

        # desired number of samples to report the average across
        self.num_samples = kwargs['samples']
        self.cur_sample = 0

        self.multiples = range(2, kwargs['end_multiple'])
        self.cur_multiple = 0
        self.cur_window = 0
        
        self.samples = {}

        self.averages = {
            'window': [],
            'multiple': [],
            'Nd1': [],
            'Nd2': [],
            'discovery_rate1': [],
            'discovery_rate2': []
        }

        if 'output_filename' in kwargs:
            self.output_filename = kwargs['output_filename']
        else:
            self.output_filename = 'multiple_window.csv'

        if 'overwrite' in kwargs:
            self.overwrite = kwargs['overwrite']

            # overwrite the old data if specified and it exists
            if self.overwrite:
                if path.exists(fn):
                    remove(fn)
    
    def update(self):
        # haven't collected the required number of samples yet...
        if self.cur_sample < self.num_samples:
            # clear the nodes' discoveries at the beginning of the sample
            if self.scene.time == self.trigger_step:
                for name, node in self.scene.nodes.items():
                    node.clear_C()

                    # init the dict for storing each node's samples if we haven't already
                    if name not in self.samples:
                        self.samples[name] = {
                            'Nd1': [],
                            'Nd2': [],
                            'discovery_rate1': [],
                            'discovery_rate2': []
                        }

            elif self.scene.time == self.trigger_step + (self.window + (self.cur_window * self.multiples[self.cur_multiple] * self.window)):
                logging.info(f'[NdiffWindowLogger] Sample {self.cur_sample} from window \
                {self.window + (self.cur_window * self.multiples[self.cur_multiple] * self.window)} (Nd{self.cur_window + 1})')
                
                # for tracking the average (across all nodes) of Nd and discovery rate
                Nd_total = 0
                rate_total = 0

                for name, node in self.scene.nodes.items():
                    Nd = node.C_cardinality()
                    rate = node.C_completeness()

                    self.samples[name][f'Nd{self.cur_window + 1}'].append(Nd)
                    self.samples[name][f'discovery_rate{self.cur_window + 1}'].append(rate)
                    
                    Nd_total += Nd
                    rate_total += rate
                
                self.averages[f'Nd{self.cur_window + 1}'].append(Nd_total / len(self.scene.nodes))
                self.averages[f'discovery_rate{self.cur_window + 1}'].append(rate_total / len(self.scene.nodes))

                # this was the first window, move on to the second window
                if self.cur_window == 0:
                    self.averages['window'].append(self.window)
                    self.averages['multiple'].append(self.multiples[self.cur_multiple])
                    node = list(self.scene.nodes.values())[0]

                    Ne = node.ndprotocol.Ne
                    Na = len(self.scene.nodes)

                    self.cur_window = 1
                # this was the second window, move onto the next sample and reset
                else:
                    assert self.cur_window == 1
                    self.cur_window = 0
                    self.cur_sample += 1
                    self.trigger_step = self.scene.time + random.randint(0, self.window*2)
                    
                    # move on to the next window multiple
                    if self.cur_sample == self.num_samples and self.cur_multiple < (len(self.multiples) - 1):
                        self.cur_sample = 0
                        self.cur_multiple += 1

                        # dump the data to csv so that we don't use a ton of memory for big evals
                        self.run_evaluation()

                        self.samples = {}
                        self.averages = {
                            'window': [],
                            'multiple': [],
                            'Nd1': [],
                            'Nd2': [],
                            'discovery_rate1': [],
                            'discovery_rate2': []
                        }

        else:
            sys.exit()

    def run_evaluation(self):
        # save data averaged across samples
        header = ['P', 'Lambda', 'Ne', 'Na', 'window', 'average_Nd', 'average_Ndiff', 'average_discovery_rate', 'multiple']

        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, 'results', 'data', self.output_filename)

        if not path.exists(fn):
            with open(fn, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a', newline='') as f:
            node = list(self.scene.nodes.values())[0]
            df = pd.DataFrame().from_dict(self.averages)
            
            for multiple in df['multiple'].unique():
                # Nd in first window
                row1 = [
                    node.ndprotocol.P,
                    node.ndprotocol.Lambda,
                    node.ndprotocol.Ne,
                    len(self.scene.nodes),
                    self.window,
                    df.loc[df['multiple'] == multiple]['Nd1'].mean(),
                    0,
                    df.loc[df['multiple'] == multiple]['discovery_rate1'].mean(),
                    multiple
                ]

                #Nd in second window
                row2 = [
                    node.ndprotocol.P,
                    node.ndprotocol.Lambda,
                    node.ndprotocol.Ne,
                    len(self.scene.nodes),
                    self.window * multiple,
                    df.loc[df['multiple'] == multiple]['Nd2'].mean(),
                    statistics.mean([Nd2 - Nd1 for Nd1, Nd2 in zip(df.loc[df['multiple'] == multiple]['Nd1'], df.loc[df['multiple'] == multiple]['Nd2'])]),
                     df.loc[df['multiple'] == multiple]['discovery_rate2'].mean(),
                    multiple
                ]

                writer = csv.writer(f)
                writer.writerow(row1)
                writer.writerow(row2)
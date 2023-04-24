import csv
import statistics
from os import path, remove
import logging
import random
import sys

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

""" Evaluates additional nodes discovered when the window size is doubled; i.e., the difference in nodes
discovered between Lambda and 2*Lambda in a subsequent run. To illustrate: 

t ----------------------------->
    [  Lambda  ][  Lambda  ]
    |   Nd1    |
    |   Nd2                |
  
    Ndiff = Nd2 - Nd1
"""
class NdiffDistributionLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # the window size (usually lambda)
        self.window = kwargs['window']

        # desired number of samples to report the average across
        self.num_samples = kwargs['samples']
        self.cur_sample = 0

        self.cur_window = 1
        
        self.samples = {}

        self.averages = {
            'Nd1': [],
            'Nd2': [],
            'discovery_rate1': [],
            'discovery_rate2': []
        }
        
        self.output_filename = kwargs['output_filename']

        if 'monitor_underperformance' in kwargs:
            self.monitor_underperformance = bool(kwargs['monitor_underperformance'])
        else:
            self.monitor_underperformance = False

        if 'overwrite' in kwargs:
            self.overwrite = kwargs['overwrite']
        else:
            self.overwrite = 0

        self.has_run = False # hackiness
    
    def update(self):
        # haven't collected the required number of samples yet...
        if self.cur_sample < self.num_samples:
            # clear the nodes' discoveries at the beginning of the sample
            if self.scene.time == self.trigger_step:
                for name, node in self.scene.nodes.items():
                    node.clear_C()

                    # clear the schedule plot so we can analyze what happened only in the windows if we need to
                    # node.clear_schedule_plot()

                    # init the dict for storing each node's samples if we haven't already
                    if name not in self.samples:
                        self.samples[name] = {
                            'Nd1': [],
                            'Nd2': [],
                            'discovery_rate1': [],
                            'discovery_rate2': []
                        }

            # reached end of current window, log Nd
            elif self.scene.time == self.trigger_step + (self.window * self.cur_window):
                logging.info(f'[NdiffDistributionLogger] Sample {self.cur_sample} from window {self.window * self.cur_window} (Nd{self.cur_window})')
                
                # for tracking the average (across all nodes) of Nd and discovery rate
                Nd_total = 0
                rate_total = 0

                for name, node in self.scene.nodes.items():
                    Nd = node.C_cardinality()
                    rate = node.C_completeness()

                    self.samples[name][f'Nd{self.cur_window}'].append(Nd)
                    self.samples[name][f'discovery_rate{self.cur_window}'].append(rate)
                    
                    Nd_total += Nd
                    rate_total += rate
                
                self.averages[f'Nd{self.cur_window}'].append(Nd_total / len(self.scene.nodes))
                self.averages[f'discovery_rate{self.cur_window}'].append(rate_total / len(self.scene.nodes))

                # this was the first window, move on to the second window
                if self.cur_window == 1:
                    # investigate any misses if the discovery rate was abnormally low...
                    node = list(self.scene.nodes.values())[0]

                    average_discovery_rate = round(self.averages[f'discovery_rate{self.cur_window}'][-1], 2)
                    promised_discovery_rate = node.ndprotocol.P
                    
                    Ne = node.ndprotocol.Ne
                    Na = len(self.scene.nodes)
                    
                    if average_discovery_rate < promised_discovery_rate and Ne == Na and self.monitor_underperformance:
                        logging.warning(f'Discovery rate on this sample beneath lower bound {promised_discovery_rate}!')
                        logging.warning(f'\tactual: {average_discovery_rate}')
                        
                        # plot all the schedules
                        plotter = NodeSchedulePlotter(trigger_step=1, nodes=','.join(self.scene.nodes.keys()))
                        plotter.set_scene(self.scene)
                        plotter.run_evaluation()

                        misses = self.scene.get_missed_discovery_names()

                        for miss in misses:
                            print(miss)
                            print(f'\t{self.scene.nodes[miss[0]].start_offset}, {self.scene.nodes[miss[1]].start_offset}')
                            plotter = NodeSchedulePlotter(trigger_step=1, nodes=f'{miss[0]},{miss[1]}')
                            plotter.set_scene(self.scene)
                            plotter.run_evaluation()

                    self.cur_window = 2
                # this was the second window, move onto the next sample and reset
                else:
                    assert self.cur_window == 2
                    self.cur_window = 1
                    self.cur_sample += 1
                    self.trigger_step = self.scene.time + random.randint(0, self.window*2)
        elif not self.has_run:
            self.run_evaluation()
            sys.exit()
            # self.scene.env_stop()
            # self.has_run = True

    def run_evaluation(self):
        # save data averaged across samples
        header = ['P', 'Lambda', 'Ne', 'Na', 'window', 'average_Nd', 'average_Ndiff', 'average_discovery_rate']

        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, 'results', 'data', self.output_filename)

        # overwrite the old data if specified and it exists
        if self.overwrite:
            if path.exists(fn):
                remove(fn)

        if not path.exists(fn):
            with open(fn, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a', newline='') as f:
            node = list(self.scene.nodes.values())[0]

            # Nd in first window
            row1 = [
                node.ndprotocol.P,
                node.ndprotocol.Lambda,
                node.ndprotocol.Ne,
                len(self.scene.nodes),
                self.window,
                statistics.mean(self.averages['Nd1']),
                0,
                statistics.mean(self.averages['discovery_rate1'])
            ]

            #Nd in second window
            row2 = [
                node.ndprotocol.P,
                node.ndprotocol.Lambda,
                node.ndprotocol.Ne,
                len(self.scene.nodes),
                self.window * 2,
                statistics.mean(self.averages['Nd2']),
                statistics.mean([Nd2 - Nd1 for Nd1, Nd2 in zip(self.averages['Nd1'], self.averages['Nd2'])]),
                statistics.mean(self.averages['discovery_rate2'])
            ]

            writer = csv.writer(f)
            writer.writerow(row1)
            writer.writerow(row2)

        # save data about the distribution of samples
        header = ['P', 'Lambda', 'Ne', 'Na', 'window', 'Ndiff', 'frequency', 'variance']

        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, 'results', 'data', f'{self.output_filename}_distribution.csv')

        # overwrite the old data if specified and it exists
        if self.overwrite:
            if path.exists(fn):
                remove(fn)

        if not path.exists(fn):
            with open(fn, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a', newline='') as f:
            # grab first node so we can get its schedule configuration (assumes all nodes have same config)
            node = list(self.scene.nodes.values())[0]

            Nd1s = [item for sublist in [node['Nd1'] for node in self.samples.values()] for item in sublist]
            Nd2s = [item for sublist in [node['Nd2'] for node in self.samples.values()] for item in sublist]
            
            Ndiffs = [Nd2 - Nd1 for Nd1, Nd2 in zip(Nd1s, Nd2s)]

            for Ndiff in set(Ndiffs):                   
                row = [
                    node.ndprotocol.P,              # P
                    node.ndprotocol.Lambda,         # Lambda
                    node.ndprotocol.Ne,             # Ne
                    len(self.scene.nodes),          # Na
                    self.window,                    # window_size
                    Ndiff,                          # diff in Nd between window and window * 2
                    Ndiffs.count(Ndiff),            # count of Ndiff
                    statistics.variance(Ndiffs)     # variance
                ]

                writer = csv.writer(f)
                writer.writerow(row)
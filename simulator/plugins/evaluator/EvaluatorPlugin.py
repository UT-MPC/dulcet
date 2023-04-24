# implements an "evaluator" which logs different data at different times during the run of the simulation
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import csv
from os import path, remove, makedirs
import sys
import logging
import random
import statistics

import numpy as np

class EvaluatorPlugin(ABC):
    def __init__(self, **kwargs):
        self.trigger_step = kwargs['trigger_step']

    def set_scene(self, scene):
        self.scene = scene

    def get_csv_writer(self, header, fn, overwrite=True, subdirectory=''):
        cur = path.dirname(path.dirname(path.abspath(__file__)))
        full_path = path.join(cur, '..', 'results', 'data', subdirectory)
        
        if not path.exists(full_path):
            makedirs(full_path)

        full_path = path.join(full_path, fn)

        # overwrite the old data if specified and it exists
        if overwrite:
            if path.exists(full_path):
                logging.warning(
                    f'[EvaluatorPlugin] Heads up: {fn} exists and will be overwritten. ' \
                    'Continue? (y/n)'
                )
                answer = input()
                if answer == 'y':
                    logging.info(f'[EvaluatorPlugin] Overwriting {fn}')
                    remove(full_path)
                else:
                    logging.info('[EvaluatorPlugin] Aborting.')
                    sys.exit()

        # write the header if the file doesn't exist
        if not path.exists(full_path):
            with open(full_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        
        return csv.writer(open(full_path, mode='a', newline=''))

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def run_evaluation(self):
        pass

""" An evaluator that plots the first node's discovery rate (i.e. completeness of C) over time.
"""
class DiscoveryRateEvaluator(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discovery_rate = []
        self.ts = []

    def update(self):
        # grab first node and log its completeness (i.e. discovery rate)
        self.discovery_rate.append(list(self.scene.nodes.values())[0].C_completeness())
        self.ts.append(self.scene.time)

        # generate the plot at the trigger step
        if self.scene.time == self.trigger_step:
            self.run_evaluation()
        
    def run_evaluation(self):
        plt.plot(self.ts, self.discovery_rate)
        plt.show()

""" An evaluator that logs the discovery rate at the trigger step in a csv file.
"""
class BLEndDiscoveryRateLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_step = kwargs['start_step']

        if 'output_filename' in kwargs:
            self.output_filename = kwargs['output_filename']
        else:
            self.output_filename = 'discovery_rate.csv'

    def update(self):
        # clear every node's discoveries at the start step
        if self.scene.time == self.start_step:
            for _, node in self.scene.nodes.items():
                node.clear_C()

        # generate the log at the trigger step
        elif self.scene.time == self.trigger_step:
            self.run_evaluation()
        
    def run_evaluation(self):
        header = ['P', 'Lambda', 'Ne', 'Na', 'start_step', 'time_elapsed', 'discovery_rate']

        # find the abs path because this is likely being invoked outside of the directory this module is in
        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, f'results/{self.output_filename}')

        # write the header if the file doesn't exist
        if not path.exists(fn):
            with open(fn, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a') as f:
            discovery_rate = 0

            # compute average discovery rate across all nodes
            for _, node in self.scene.nodes.items():
                discovery_rate += node.C_completeness()

            discovery_rate = discovery_rate / len(self.scene.nodes)

            # grab first node so we can get its schedule configuration
            node = list(self.scene.nodes.values())[0]

            row = [
                node.blend_parameters.P,        # P
                node.blend_parameters.Lambda,   # Lambda
                node.blend_parameters.N,        # Ne
                len(self.scene.nodes),          # Na
                self.start_step,                # step when we started measuring discovery rate
                self.scene.time,                # scene time
                discovery_rate                  # average discovery rate
            ]

            writer = csv.writer(f)
            writer.writerow(row)

            sys.exit()

""" An evaluator that logs the time the average discovery rate hit P (in a csv file)
"""
class BLEndDiscoveryLatencyLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(trigger_step=None)
        self.start_step = kwargs['start_step']

        if 'output_filename' in kwargs:
            self.output_filename = kwargs['output_filename']
        else:
            self.output_filename = 'discovery_latency.csv'

    def update(self):
        # clear every node's discoveries at the start step
        if self.scene.time == self.start_step:
            for _, node in self.scene.nodes.items():
                node.clear_C()

        if self.scene.time >= self.start_step:
            discovery_rate = 0

            # compute the current average discovery rate
            for _, node in self.scene.nodes.items():
                discovery_rate += node.C_completeness()

            # check if the average discovery rate meets or exceeds P, and that we haven't already fired off the logger (we only want to fire it the first time we hit P)
            if discovery_rate / len(self.scene.nodes) >= list(self.scene.nodes.values())[0].blend_parameters.P and self.trigger_step == None:
                self.trigger_step = self.scene.time
                self.run_evaluation()


    def run_evaluation(self):
        header = ['P', 'Lambda', 'Ne', 'Na', 'discovery_latency']

        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, f'results/{self.output_filename}')

        # write the header if the file doesn't exist
        if not path.exists(fn):
            with open(fn, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a') as f:
            # grab first node so we can get its schedule configuration
            node = list(self.scene.nodes.values())[0]

            row = [
                node.blend_parameters.P,        # P
                node.blend_parameters.Lambda,   # Lambda
                node.blend_parameters.N,        # Ne
                len(self.scene.nodes),          # Na
                self.scene.time - self.start_step  # current scene time minus start time, which is how long it took for discovery rate to hit 100%
            ]

            writer = csv.writer(f)
            writer.writerow(row)

            sys.exit()

""" An evaluator that logs the average discovery rate at each time step
"""
class OldContinuousDiscoveryRateLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discovery_rates = []
        self.ts = []

    def update(self):
        discovery_rate = 0

         # compute the current average discovery rate
        for _, node in self.scene.nodes.items():
            discovery_rate += node.C_completeness()

        discovery_rate = discovery_rate / len(self.scene.nodes)

        self.discovery_rates.append(discovery_rate)
        self.ts.append(self.scene.time)

        # append csv at trigger step
        if self.scene.time == self.trigger_step:
            self.run_evaluation()

    def run_evaluation(self):
        # grab first node so we can get its schedule configuration
        node = list(self.scene.nodes.values())[0]

        header = ['P', 'Lambda', 'Ne', 'Na', 'time_elapsed', 'discovery_rate']

        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, f'results/continuous_discovery_rate_P={node.blend_parameters.P}-Lambda={node.blend_parameters.Lambda}.csv')

        # write the header if the file doesn't exist
        if not path.exists(fn):
            with open(fn, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a') as f:
            for i, discovery_rate in enumerate(self.discovery_rates):
                row = [
                    node.blend_parameters.P,        # P
                    node.blend_parameters.Lambda,   # Lambda
                    node.blend_parameters.N,        # Ne
                    len(self.scene.nodes),          # Na
                    self.ts[i],                     # scene time
                    discovery_rate                  # average discovery rate
                ]

                writer = csv.writer(f)
                writer.writerow(row)

""" An evaluator that plots the energy consumption over time of the first node.
"""
class EnergyConsumptionPlotter(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.energy_consumptions = []
        self.unadapted_energy_consumptions = []
        self.ts = []

    def update(self):
        node = list(self.scene.nodes.values())[0]

        if not hasattr(self, 'init_Q'):
            # need to track the initial Q and initial epoch
            # since Q is per epoch, we have to know the epoch length we started with to be able
            # to calculate what the energy draw would have been had we stuck with that schedule
            self.init_Q = node.blend_parameters.Q
            self.init_E = node.blend_parameters.epoch
            self.unadapted_energy_consumptions.append(0)
            self.last_epoch = self.scene.time

        # an epoch has passed
        if self.scene.time == self.last_epoch + self.init_E:
            self.unadapted_energy_consumptions.append(self.unadapted_energy_consumptions[-1] + self.init_Q)
            self.last_epoch = self.scene.time
        elif self.scene.time != 3:
            self.unadapted_energy_consumptions.append(self.unadapted_energy_consumptions[-1])

        self.ts.append(self.scene.time)
        self.energy_consumptions.append(node.blend_parameters.energy_consumption)

        if self.scene.time == self.trigger_step:
            self.run_evaluation()

    def run_evaluation(self):
        fig, ax = plt.subplots()

        ax.plot(self.ts, self.energy_consumptions, label='Adaptive')
        ax.plot(self.ts, self.unadapted_energy_consumptions, label='Static')

        for t in list(self.scene.nodes.values())[0].reoptimization_times:
            plt.axvline(t[0], ls='--', color='dimgray')

        plt.title('Energy consumption of BLEnd over time')
        plt.xlabel('time (ms)')
        plt.ylabel('total instantaneous current draw (mAh)')
        plt.legend()
        plt.show()

        # the lifetime of the protocol given the battery is the mAh of the battery divided by the protocol's instananeous current
        battery = 225 * 3 / 2.1
        print(f'lifetime: {battery/self.energy_consumptions[-1]}')

""" An evaluator that plots the number of collisions at each time step.
"""
class CollisionPlotter(EvaluatorPlugin):
    def update(self):
        if self.scene.time == self.trigger_step:
            self.run_evaluation()

    def run_evaluation(self):
        fig, ax = plt.subplots()

        pts = list(map(list, zip(*self.scene.collisions)))

        ax.plot(pts[0], pts[1])

        plt.title('Beacon collisions over time')
        plt.xlabel('time (ms)')
        plt.ylabel('number of collisions')
        plt.show()     

""" Exits the sim when the given scanner node first discovers the given advertiser
"""
class Debugger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.advertiser = kwargs['advertiser']
        self.scanner = kwargs['scanner']

    def update(self):
        node = self.scene.nodes[self.scanner]

        if node.has_C_neighbor(self.advertiser):
            self.run_evaluation()
        elif self.scene.time == self.trigger_step:
            print(f'[{self.scene.time}] Node {self.scanner} did not discover neighbor!')
            sys.exit()

    def run_evaluation(self):
        print(f'[{self.scene.time}] Node {self.scanner} discovered neighbor {self.advertiser}')
        sys.exit()

""" Evaluates the difference in discovered nodes as we increase the window size for discovery
beyond lambda. Does so by taking n randomly-spaced samples of Nd and discovery rate in 
window + 0*increment, then taking n randomly-spaced samples of Nd and discovery rate in 
window + 1*increment, and so on.

t -------------------------------------------------------------->
    [  window  ]     ...      [  window+(increment*i)        ]
    |   Nd1    |
                              |   Nd2                        |
    Ndiff = Nd2 - Nd1

Can we maybe determine how many Na nodes there are based on the rate of change as the
window size increases?
"""
class IncrementalWindowEvaluator(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # the window size (usually lambda)
        self.window = kwargs['window']

        # how much to add to the window each time
        self.increment = kwargs['increment']

        # how many times to increment the window
        self.multiples = kwargs['multiples']

        # a list of the actual amounts to add to the window
        self.increments = [i * self.increment for i in range(0, self.multiples+1)]

        # the current increment in the list we're evaluating (starting with the first at zero)
        self.current_increment = 0

        # desired number of samples to report the average across
        self.num_samples = kwargs['samples']

        # lists for samples and averaged discovery rates across those samples
        self.samples = []
        self.discovery_rates = []

        if 'output_filename' in kwargs:
            self.output_filename = kwargs['output_filename']
        else:
            self.output_filename = 'incremental_window.csv'

        if 'output_distribution' in kwargs:
            self.output_distribution = bool(kwargs['output_distribution'])
        else:
            self.output_distribution = False

        self.distribution = {}

    def update(self):
        # starting at trigger_step, begin randomly sampling the discovery rate within the window
        if self.scene.time == self.trigger_step:
            for _, node in self.scene.nodes.items():
                node.clear_C()
        # window has passed, sample the discovery rate and reset the trigger step
        elif self.scene.time == self.trigger_step + self.window + self.increments[self.current_increment]:
            logging.info(f'[IncrementalWindowEvaluator] Increment {self.current_increment} sample {len(self.samples)}')
            discovery_rate = 0
            Nd = 0

            # compute average discovery rate across all nodes
            for _, node in self.scene.nodes.items():
                discovery_rate += node.C_completeness() # what percent of the total actual neighbors have been discovered
                Nd += node.C_cardinality()              # how many neighbors have been discovered

                if self.output_distribution:
                    current_window = self.window + self.increments[self.current_increment]
                    
                    if current_window in self.distribution:
                        self.distribution[current_window].append(node.C_cardinality())
                    else:
                        self.distribution[current_window] = [node.C_cardinality()]
                        

            discovery_rate = discovery_rate / len(self.scene.nodes) # average discovery rate across all nodes
            Nd = Nd / len(self.scene.nodes)                         # average number of nodes discovered across all nodes
            
            self.samples.append((discovery_rate, Nd))

            # if we have gathered the desired number of samples, 
            # collect the mean of the samples, record the value in the csv, 
            # reset the samples, and move to the next increment
            if len(self.samples) == self.num_samples:
                header = ['P', 'Lambda', 'Ne', 'Na', 'window_size', 'Nd', 'Nd_variance', 'discovery_rate', 'discovery_rate_variance']

                cur = path.dirname(path.dirname(path.abspath(__file__)))
                fn = path.join(cur, f'results/{self.output_filename}')

                # write the header if the file doesn't exist
                if not path.exists(fn):
                    with open(fn, mode='w') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                with open(fn, mode='a') as f:
                    # grab first node so we can get its schedule configuration (assumes all nodes have same config)
                    node = list(self.scene.nodes.values())[0]

                    row = [
                        node.blend_parameters.P,        # P
                        node.blend_parameters.Lambda,   # Lambda
                        node.blend_parameters.N,        # Ne
                        len(self.scene.nodes),          # Na
                        self.window + self.increments[self.current_increment], # window_size
                        statistics.mean([x[1] for x in self.samples]), # Nd averaged across samples
                        statistics.variance([x[1] for x in self.samples]), # variance in Nd
                        statistics.mean([x[0] for x in self.samples]),  # discovery rate averaged across samples
                        statistics.variance([x[0] for x in self.samples]) # variance in discovery rate
                    ]

                    writer = csv.writer(f)
                    writer.writerow(row)

                self.samples = []
                self.current_increment += 1

            # run the eval if we have gathered data for all increments
            if self.current_increment == len(self.increments):
                self.run_evaluation()
                sys.exit()
            
            # pick a new start time to begin sampling
            self.trigger_step = self.scene.time + random.randint(0, self.window*2)

    def run_evaluation(self):
        if self.output_distribution:
            print(self.distribution)
            header = ['P', 'Lambda', 'Ne', 'Na', 'window_size', 'Nd', 'frequency']

            cur = path.dirname(path.dirname(path.abspath(__file__)))
            fn = path.join(cur, f'results/{self.output_filename}_distribution.csv')

            if not path.exists(fn):
                with open(fn, mode='w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

            with open(fn, mode='a') as f:
                # grab first node so we can get its schedule configuration (assumes all nodes have same config)
                node = list(self.scene.nodes.values())[0]

                for window in self.distribution.keys():
                    Nds = set(self.distribution[window])    # turn into set to get unique Nd

                    for Nd in Nds:                      
                        row = [
                            node.blend_parameters.P,        # P
                            node.blend_parameters.Lambda,   # Lambda
                            node.blend_parameters.N,        # Ne
                            len(self.scene.nodes),          # Na
                            window,                         # window_size
                            Nd,                             # Nd in window
                            self.distribution[window].count(Nd) # count of occurences of Nd
                        ]

                        writer = csv.writer(f)
                        writer.writerow(row)

""" Evaluates additional nodes discovered when the window size is doubled; i.e., the difference in nodes
discovered between Lambda and 2*Lambda in a subsequent run. To illustrate: 

t ----------------------------->
    [  Lambda  ][  Lambda  ]
    |   Nd1    |
    |   Nd2                |
  
    Ndiff = Nd2 - Nd1
"""
class OldDoubleWindowEvaluator(EvaluatorPlugin):
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

        if 'output_filename' in kwargs:
            self.output_filename = kwargs['output_filename']
        else:
            self.output_filename = 'double_window.csv'

        if 'monitor_underperformance' in kwargs:
            self.monitor_underperformance = bool(kwargs['monitor_underperformance'])
        else:
            self.monitor_underperformance = False
    
    def update(self):
        # haven't collected the required number of samples yet...
        if self.cur_sample < self.num_samples:
            # clear the nodes' discoveries at the beginning of the sample
            if self.scene.time == self.trigger_step:
                for name, node in self.scene.nodes.items():
                    node.clear_C()

                    # clear the schedule plot so we can analyze what happened only in the windows if we need to
                    node.clear_schedule_plot()

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
                logging.info(f'[DoubleWindowEvaluator] Sample {self.cur_sample} from window {self.window * self.cur_window} (Nd{self.cur_window})')
                
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
                    promised_discovery_rate = node.blend_parameters.P
                    
                    Ne = node.blend_parameters.N
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
        else:
            self.run_evaluation()
            sys.exit()

    def run_evaluation(self):
        # save data averaged across samples
        header = ['P', 'Lambda', 'Ne', 'Na', 'window', 'average_Nd', 'average_Ndiff', 'average_discovery_rate']

        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, f'results/{self.output_filename}')

        if not path.exists(fn):
            with open(fn, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a') as f:
            node = list(self.scene.nodes.values())[0]

            # Nd in first window
            row1 = [
                node.blend_parameters.P,
                node.blend_parameters.Lambda,
                node.blend_parameters.N,
                len(self.scene.nodes),
                self.window,
                statistics.mean(self.averages['Nd1']),
                0,
                statistics.mean(self.averages['discovery_rate1'])
            ]

            #Nd in second window
            row2 = [
                node.blend_parameters.P,
                node.blend_parameters.Lambda,
                node.blend_parameters.N,
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
        fn = path.join(cur, f'results/{self.output_filename}_distribution.csv')

        if not path.exists(fn):
            with open(fn, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        with open(fn, mode='a') as f:
            # grab first node so we can get its schedule configuration (assumes all nodes have same config)
            node = list(self.scene.nodes.values())[0]

            Nd1s = [item for sublist in [node['Nd1'] for node in self.samples.values()] for item in sublist]
            Nd2s = [item for sublist in [node['Nd2'] for node in self.samples.values()] for item in sublist]
            
            Ndiffs = [Nd2 - Nd1 for Nd1, Nd2 in zip(Nd1s, Nd2s)]

            for Ndiff in set(Ndiffs):                   
                row = [
                    node.blend_parameters.P,        # P
                    node.blend_parameters.Lambda,   # Lambda
                    node.blend_parameters.N,        # Ne
                    len(self.scene.nodes),          # Na
                    self.window,                    # window_size
                    Ndiff,                          # diff in Nd between window and window * 2
                    Ndiffs.count(Ndiff),            # count of Ndiff
                    statistics.variance(Ndiffs)     # variance
                ]

                writer = csv.writer(f)
                writer.writerow(row)

# from Scene import Scene
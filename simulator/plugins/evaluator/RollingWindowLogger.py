from os import path, remove
import sys
import csv
from statistics import mean, stdev
import logging

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

BATCH_SIZE = 1000

# logs percent discoveries made in the window [t - lambda, t] at each time t
# in mobile simulations, the discovery rate reflects the percentage of actual neighbors 
# discovered within each node's collision domain.
class RollingWindowLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discovery_rates = []
        self.ts = []
        self.Nas = []

        output_filename = kwargs['output_filename']

        if 'Lambda' in kwargs:
            self.Lambda = kwargs['Lambda']
        else:
            self.Lambda = None

        # if a reference node is given, report stats about only that node; otherwise report averaged across all nodes
        if 'reference_node' in kwargs:
            self.reference_node = kwargs['reference_node']
            self.report_all_nodes = False

            output_filename = self.reference_node + '_' + output_filename
        elif 'report_all_nodes' in kwargs:
            self.reference_node = None
            self.report_all_nodes = bool(int(kwargs['report_all_nodes']))
        else:
            self.reference_node = None
            self.report_all_nodes = False

        header = ['t', 'node', 'discovery_rate', 'Na', 'average_Na', 'average_Ne', 'pos_x', 'pos_y', 'min_discovery_rate', 'max_discovery_rate', 'stdev_discovery_rate']
        self.writer = self.get_csv_writer(header, output_filename, subdirectory='discovery-stats')

        self.batch = []

    def discovery_count(self, node, window):
        discoveries = 0

        for k, d in node.C.items():
            # if the discovery time falls in the window 
            # AND the discovered neighbor is still in the collision domain, log it
            if d[0] <= self.scene.time and d[0] >= self.scene.time - window and k in node.G.keys() and k in self.scene.nodes:
                discoveries += 1
                
        return discoveries

    def single_node_stats(self, node):
        discovery_rates = []
        average_Na = 0
        average_Ne = 0
        discoveries = 0
        
        if self.Lambda is not None:
            window = self.Lambda
        else:
            window = node.ndprotocol.Lambda

        discoveries = self.discovery_count(node, window)
        Na = len(node.G) + 1

        if Na > 1:
            discovery_rates.append(discoveries / (Na - 1))
        else:
            discovery_rates.append(None)

        average_Na += (Na - 1)
        average_Ne += node.ndprotocol.Nep

        pos_x = node.position[0]
        pos_y = node.position[1]

        discovery_rates = [x for x in discovery_rates if x is not None]

        if len(discovery_rates) > 0:
            discovery_rate = mean(discovery_rates)
        else:
            discovery_rate = None

        row = [
            self.scene.time,        # scene time
            node.name,              # name of node
            discovery_rate,         # average discovery rate
            len(self.scene.nodes),  # Na
            average_Na/len(self.scene.nodes) if self.reference_node == None else average_Na, # average number of actual neighbors
            average_Ne/len(self.scene.nodes) if self.reference_node == None else average_Ne, # average estimated number of neighbors
            pos_x,
            pos_y,
            None,
            None,
            None
        ]

        return row

    def aggregate_stats(self):
        discovery_rates = []
        average_Na = 0
        average_Ne = 0

        # log how many discoveries were made in [t - Lambda, t]
        for _, node in self.scene.nodes.items():
            discoveries = 0
            
            if self.Lambda is not None:
                window = self.Lambda
            else:
                window = node.ndprotocol.Lambda

            discoveries = self.discovery_count(node, window)
            Na = len(node.G) + 1

            if Na > 1:
                discovery_rates.append(discoveries / (Na - 1))
            else:
                discovery_rates.append(None)

            average_Na += (Na - 1)
            average_Ne += node.ndprotocol.Nep
            
        # no position is given when reporting stats across all nodes
        pos_x = None
        pos_y = None

        if len(discovery_rates) > 0:
            discovery_rates = [x for x in discovery_rates if x is not None]
            discovery_rate = mean(discovery_rates)
        else:
            discovery_rate = None

        row = [
            self.scene.time,        # scene time
            None,                   # none, since these are stats for all nodes
            discovery_rate,         # average discovery rate
            len(self.scene.nodes),  # Na
            average_Na/len(self.scene.nodes) if self.reference_node == None else average_Na, # average number of actual neighbors
            average_Ne/len(self.scene.nodes) if self.reference_node == None else average_Ne, # average estimated number of neighbors
            pos_x,
            pos_y,
            min(discovery_rates),
            max(discovery_rates),
            stdev(discovery_rates)
        ]

        return row

    def batch_row(self, row):
        self.batch.append(row)

        if len(self.batch) >= BATCH_SIZE:
            self.writer.writerows(iter(self.batch))
            self.batch = []

    def update(self):
        # it's IMPORTANT that the trigger step is set to something nonzero, some time
        # after all the nodes have had a chance to start advertising (i.e., after the "warmup" period).
        # otherwise it will look like the schedule is underperforming – that's just because not everyone 
        # has started their first epoch yet
        if self.scene.time == self.trigger_step:
            if self.reference_node is None:
                for _, node in self.scene.nodes.items():
                    node.clear_C()
            else:
                self.scene.nodes[self.reference_node].clear_C

        if self.scene.time >= self.trigger_step:
            # report the average discovery rate across all nodes
            if self.reference_node is None and not self.report_all_nodes:
                row = self.aggregate_stats()
                self.batch_row(row)
                return
            
            # or, report the discovery rate for each node
            elif self.report_all_nodes:
                for node in self.scene.nodes.values():
                    row = self.single_node_stats(node)
                    self.batch_row(row)
                return

            # otherwise, report discovery rate for the focus node in addition to the aggregate stats
            else:
                row1 = self.single_node_stats(self.scene.nodes[self.reference_node])
                row2 = self.aggregate_stats()

                self.batch_row(row1)
                self.batch_row(row2)
                return

    def run_evaluation(self):
        pass
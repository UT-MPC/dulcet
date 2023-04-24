from os import path, remove
import csv

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

""" An evaluator that logs the average discovery rate at each time step
"""
class ContinuousDiscoveryRateLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discovery_rates = []
        self.ts = []
        self.Nas = []

        header = ['t', 'discovery_rate', 'Na']
        self.writer = self.get_csv_writer(header, kwargs['output_filename'])

    def update(self):
        # it's IMPORTANT that the trigger step is set to something nonzero, some time
        # after all the nodes have had a chance to start advertising (i.e., after the "warmup" period).
        # otherwise it will look like the schedule is underperforming – that's just because not everyone 
        # has started their first epoch yet
        if self.scene.time == self.trigger_step:
            for _, node in self.scene.nodes.items():
                node.clear_C()

        if self.scene.time >= self.trigger_step:
            discovery_rate = 0

            # compute the current average discovery rate
            for _, node in self.scene.nodes.items():
                discovery_rate += node.C_completeness()

            Na = len(self.scene.nodes)
            discovery_rate = discovery_rate / Na

            row = [
                self.scene.time,      # scene time
                discovery_rate,       # average discovery rate
                Na                    # Na
            ]

            self.writer.writerow(row)

    def run_evaluation(self):
        pass
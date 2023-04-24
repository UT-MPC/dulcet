from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

from os import path, remove
import csv

""" Logs the overall error (across all nodes) in estimate over time
"""
class AdaptationErrorLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(trigger_step=kwargs['trigger_step'])

        # step size between measurements
        self.step_size = kwargs['step_size']

        header = ['t', 'Na', 'Ne', 'Nep', 'epsilon']
        self.writer = self.get_csv_writer(header, kwargs['output_filename'])

    def update(self):
        # log it every 100 steps after the trigger step
        if self.scene.time % self.step_size == 0 and self.scene.time >= self.trigger_step:
            Na = len(self.scene.nodes)
            Ne = 0
            Nep = 0
            epsilon = 0

            for node in self.scene.nodes.values():
                Ne = node.ndprotocol.Ne
                Nep += (node.ndprotocol.Nep)
                epsilon += (Na - node.ndprotocol.Nep)

            row = [
                self.scene.time,       # t
                len(self.scene.nodes), # Na
                Ne,                    # Ne
                Nep/Na,                # average Ne prime (current estimate)
                epsilon/Na             # average error
            ]

            print(Nep/Na)

            self.writer.writerow(row)

    def run_evaluation(self):
        pass
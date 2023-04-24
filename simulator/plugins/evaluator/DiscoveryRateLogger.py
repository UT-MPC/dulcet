from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

from os import path
import csv

class DiscoveryRateLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_step = kwargs['start_step']
        self.output_filename = kwargs['output_filename']

    def update(self):
        # clear every node's discoveries at the start step
        if self.scene.time == self.start_step:
            for _, node in self.scene.nodes.items():
                node.clear_C()

        # generate the log at the trigger step
        elif self.scene.time == self.trigger_step:
            self.run_evaluation()
        
    def run_evaluation(self):
        header = ['Ne', 'Na', 'start_step', 'time_elapsed', 'discovery_rate']
        writer = self.get_csv_writer(header, self.output_filename)

        discovery_rate = 0

        # compute average discovery rate across all nodes
        for _, node in self.scene.nodes.items():
            discovery_rate += node.C_completeness()

        discovery_rate = discovery_rate / len(self.scene.nodes)

        # grab first node so we can get its schedule configuration
        node = list(self.scene.nodes.values())[0]

        row = [
            node.ndprotocol.Ne,                 # Ne
            len(self.scene.nodes),              # Na
            self.start_step,                    # step when we started measuring discovery rate
            self.scene.time - self.start_step,  # time elapsed
            discovery_rate                      # average discovery rate
        ]

        writer.writerow(row)
        node.scene.env_stop()
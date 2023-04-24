from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

from os import path
import csv

""" An evaluator that logs the time the average discovery rate hit P (in a csv file)
"""
class DiscoveryLatencyLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(trigger_step=None)
        self.start_step = kwargs['start_step']
        self.target_discovery_rate = kwargs['target_discovery_rate']
        self.output_filename = kwargs['output_filename']

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
            if discovery_rate / len(self.scene.nodes) >= self.target_discovery_rate and self.trigger_step == None:
                self.trigger_step = self.scene.time
                self.run_evaluation()

    def run_evaluation(self):
        header = ['target_discovery_rate', 'Ne', 'Na', 'discovery_latency']
        writer = self.get_csv_writer(header, self.output_filename)

        # grab first node so we can get its schedule configuration
        node = list(self.scene.nodes.values())[0]

        row = [
            self.target_rate,                   # the discovery rate hit when the latency was recorded
            node.ndprotocol.Ne,                 # Ne
            len(self.scene.nodes),              # Na
            self.scene.time - self.start_step   # current scene time minus start time, which is how long it took for discovery rate to hit the target
        ]

        writer.writerow(row)
        node.scene.env_stop()
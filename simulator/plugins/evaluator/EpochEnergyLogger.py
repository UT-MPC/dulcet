from os import path, remove
import sys
import csv
import logging

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

from protocols.blend.BLEnd import BLEnd
from protocols.birthday.Birthday import Birthday

# logs the energy consumed by each epoch of neighbor discovery
class EpochEnergyLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # track the last epoch of each node, so we only update the data when they are on a new epoch
        self.last_epoch = {}

        header = ['t', 'epoch', 'node', 'P', 'Lambda', 'Q', 'Ne', 'Na']
        self.writer = self.get_csv_writer(header, kwargs['output_filename'])

    def update(self):
        if self.scene.time >= self.trigger_step:
            # log the schedule stats (including Q) for each node on a new epoch
            for name, node in self.scene.nodes.items():
                current_epoch = node.ndprotocol.current_epoch

                if name not in self.last_epoch:
                    self.last_epoch[name] = current_epoch

                # node's neighbor discovery has moved to the next epoch, log data for its current schedule config
                if current_epoch != self.last_epoch[name]:
                    self.last_epoch[name] = current_epoch
            
                    if isinstance(node.ndprotocol, BLEnd):
                        row = [
                            self.scene.time,
                            current_epoch,
                            name,
                            node.ndprotocol.parameters.P,
                            node.ndprotocol.parameters.Lambda,
                            node.ndprotocol.parameters.Q,
                            node.ndprotocol.Nep,
                            len(node.G)
                        ]
                    elif isinstance(node.ndprotocol, Birthday):
                        row = [
                            self.scene.time,
                            current_epoch,
                            name,
                            node.ndprotocol.P,
                            node.ndprotocol.Lambda,
                            node.ndprotocol.Q,
                            node.ndprotocol.Nep,
                            len(node.G)
                        ]

                    self.writer.writerow(row)

    def run_evaluation(self):
        pass
import random
import logging

from plugins.evaluator.EvaluatorPlugin import EvaluatorPlugin

class NdaLogger(EvaluatorPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        header = ['Ne', 'Na', 'Nda1', 'Nda2', 'node']
        self.writer = self.get_csv_writer(header, kwargs['output_filename'], overwrite=False)
        self.w = kwargs['window']
        self.w_mult = kwargs['window_multiplier']
        self.Nda = {}

    def update(self):
        # clear samples at the start of the window
        if self.scene.time == self.trigger_step:
            for name, node in self.scene.nodes.items():
                node.clear_C()
        
        # collect Nda (nodes discovered in first window) after w
        elif self.scene.time == self.trigger_step + self.w:
            for name, node in self.scene.nodes.items():
                self.Nda[name] = node.C_cardinality()
                logging.info(f'[NdiffLogger:{name}:{self.scene.time}] Sample Nda = {self.Nda[name]}')
        
        # collect Nd2 and compute Ndiff after w_mult * w
        elif self.scene.time == self.trigger_step + (self.w * self.w_mult):   
            for name, node in self.scene.nodes.items():
                Nda2 = node.C_cardinality()
                logging.info(f'[NdiffLogger:{name}:{self.scene.time}] Sample Nda2 = {Nda2}')
                
                self.writer.writerow([
                    node.ndprotocol.Nep, # Ne
                    len(node.G),         # Na
                    self.Nda[name],      # Nda1
                    Nda2,                # Nda2
                    name                 # node
                ])

            self.trigger_step = random.randint(self.scene.time + 1, self.scene.time + 8001)

    def run_evaluation(self):
        pass

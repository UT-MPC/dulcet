import random
import logging
from math import ceil
from statistics import mean, variance

from protocols.adaptive.AdaptationProtocol import AdaptationProtocol

class CANDor(AdaptationProtocol):
    def __init__(self, **kwargs):
        if 'sense_only' in kwargs:
            self.sense_only = kwargs['sense_only']
        else:
            self.sense_only = False
        
        if 'w' in kwargs:
            self.w = kwargs['w']
        else:
            self.w = -1

        if 'w_mult' in kwargs:
            self.w_mult = kwargs['w_mult']
        else:
            self.w_mult = 2

        if 'si_samples' in kwargs:
            self.si_samples = kwargs['si_samples']
        else:
            self.si_samples = 10

        if 'adaptation_start_offset' in kwargs:
            self.adaptation_start_offset = kwargs['adaptation_start_offset']
        else:
            self.adaptation_start_offset = 0

        if 'tau_mean' in kwargs:
            self.tau_mean = kwargs['tau_mean']
        else:
            self.tau_mean = 0

        if 'tau_variance' in kwargs:
            self.tau_variance = kwargs['tau_variance']
        else:
            self.tau_variance = 10

        self.sample_start = self.adaptation_start_offset
        self.si = []    # si, the running sequence of ndiffa samples
        self.Ndw = -1   # nodes discovered in first window
        self.Nd2w = -1  # nodes discovered in second window

        self.U = {}

    def update_U_neighbor(self, t, j, c):
        self.U[j] = (t, c)

    def clear_U(self):
        self.U = {}

    def U_cardinality(self):
        return len(self.U.keys())

    def adapt_schedule(self):
        # clear samples at the start of the window
        if self.parent.scene.time == self.sample_start:
            self.clear_U()
        
        # collect Ndw (nodes discovered in first window) after w
        elif self.parent.scene.time == self.sample_start + self.w:
            self.Ndw = self.U_cardinality() + 1     # +1 to include ourself
            
            logging.debug(f'[Node:{self.parent.name}:{self.parent.scene.time}] Sample {len(self.si)} Ndw = {self.Ndw}')
        
        # collect Nd2 and compute Ndiff after w_mult * w
        elif self.parent.scene.time == self.sample_start + (self.w * self.w_mult):   
            self.Nd2w = self.U_cardinality() + 1    # +1 to include ourself
            
            logging.debug(f'[Node:{self.parent.name}:{self.parent.scene.time}] Sample {len(self.si)} Nd2w = {self.Nd2w}')
            
            self.si.append(self.Nd2w - self.Ndw)
            
            # have the desired number of samples, take a look at them
            if len(self.si) == self.si_samples:
                Ndiffa = mean(self.si)
                Ndiffa_var = variance(self.si)

                logging.info(f'[Node:{self.parent.name}][{self.parent.position[0]},{self.parent.position[1]}] mean(Ndiffa) = {Ndiffa}, Ndiff_var = {Ndiffa_var}')

                if (Ndiffa_var <= self.tau_variance and Ndiffa <= self.tau_mean):
                # if (Ndiffa_var <= self.tau_variance):
                    logging.info(f'[Node:{self.parent.name}][{self.parent.position[0]},{self.parent.position[1]}] Sensing overestimate based on {Ndiffa_var} <= {self.tau_variance}')
                    
                    if self.Nd2w == 2:
                        Na = self.Nd2w
                    else:
                        # round up to nearest five (difference between sched for Ne = n and Ne = n +/- 1 is minimal, if any)
                        Na = ceil(self.Nd2w / 5) * 5
                else:
                    logging.info(f'[Node:{self.parent.name}][{self.parent.position[0]},{self.parent.position[1]}] Comparing Ndiffa and Ndiff')
                    Na = self.parent.ndprotocol.estimate_Na_from_Ndiffa(Ndiffa)

                logging.info(f'[Node:{self.parent.name}][{self.parent.position[0]},{self.parent.position[1]}] Na is {Na}? (actually {len(self.parent.G) + 1})')
                self.si = []

                if Na != self.parent.ndprotocol.Ne and Na > 0:
                    if not self.sense_only:
                        self.parent.ndprotocol.adapt_schedule(Na)
                    else:
                        self.parent.ndprotocol.Nep = Na
            
            self.sample_start = self.parent.scene.time + random.randint(0, 10*self.parent.ndprotocol.Lambda)
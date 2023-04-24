# implements a continuous neighbor discovery configuration
from math import floor
import logging
import ast

import protocols.blend.BLEndModel
import protocols.blend.OptimizerDriver
import protocols.blend.ndiffer as ndiffer

class BLEndParameters:
    def __init__(self, is_ublend: bool, P: float, Lambda: int, N: int, b: int, s: int, E: int, A: int, nb: int, Q: float=0.0, P_expected: float=-1):
        self.is_ublend = is_ublend
        
        # application requirements
        self.P = P
        self.Lambda = Lambda
        self.N = N
        
        # hardware attributes
        self.beacon_duration = b
        self.max_random_slack = s

        # neighbor discovery parameters
        self.epoch = E
        self.advertising_interval = A

        # derived parameters
        self.scan_interval = A + b + s

        # u-blend
        if is_ublend:
            self.num_advertisements = floor(E/(2*A)) - 1
        # f-blend
        else:
            self.num_advertisements = floor(E/A) - 1

        if self.num_advertisements != nb:
            logging.warning(f'Input nb = {nb} != num_advertisements = {self.num_advertisements}')

        self.Q = Q
        self.energy_consumption = Q

    def optimize(self, P: float, Lambda: int, N: int):
        c = protocols.blend.OptimizerDriver.optimize(P, Lambda, N, self.beacon_duration, self.max_random_slack, self.is_ublend)
        
        # update requirements
        self.P = P
        self.Lambda = Lambda
        self.N = N

        # update schedule configuration
        self.epoch = c['E']
        self.advertising_interval = c['A']
        self.scan_interval = c['L']
        self.num_advertisements = c['nb']
        
        # update energy cost
        self.Q = c['Q']

    @classmethod
    def from_optimizer(self, P: float, Lambda: int, N: int, beacon_duration: int, max_random_slack: int, is_ublend: bool=False):
        """ Initialize a set of BLEnd parameters from a set of application requirements rather than a predetermined schedule 
        """
        c = protocols.blend.OptimizerDriver.optimize(P, Lambda, N, beacon_duration, max_random_slack, is_ublend)

        return self(c['is_ublend'], c['P'], c['Lambda'], c['N'], c['b'], c['s'], c['E'], c['A'], c['nb'], c['Q'], c['P_expected'])

    def update_energy_consumption(self):
        """ Increments the power consumed by BLEnd using the current Q value. This should be called
        at the end of an epoch since Q is the instantaneous current draw per epoch.
        """
        self.energy_consumption += self.Q

    def compute_window_for_target_P(self, P, Ne):
        """ Uses the BLEnd model to compute and return the window size required to attain the target 
        discovery probability P for the given estimated number of nodes Ne using the current schedule.
        """
        window = 0
        Pc = 0.0

        # FIXME this could hang under some circumstances where a window that provides target P does not exist; need an exit condition
        while round(Pc, 3) != P:
            window += 100
            Pc = protocols.blend.BLEndModel.compute_Pd(Ne, window, self.epoch, self.max_random_slack, self.scan_interval, self.beacon_duration, self.num_advertisements)

        return window
    
    # FIXME this could just be an actual dict instead of bringing in a whole library for this one lazy line
    def as_dict(self):
        return ast.literal_eval(self.__str__())

    def __str__(self):
        return f'''{{\"P\": {self.P}, \"Lambda\": {self.Lambda}, \"N\": {self.N}, \"b\": {self.beacon_duration}, 
        \"s\": {self.max_random_slack}, \"E\": {self.epoch}, \"A\": {self.advertising_interval}, \"L\": {self.scan_interval},
        \"Q\": {self.Q}, \"nb\": {self.num_advertisements}}}'''
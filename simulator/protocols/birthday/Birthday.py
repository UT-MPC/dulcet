from math import floor
from Events import *
import logging
import random
import math

from protocols.NeighborDiscoveryProtocol import NeighborDiscoveryProtocol

# current cost of each mode
I_SCAN = 6.329
I_ADV = 5.725
I_IDLE = 0.08064

class Birthday(NeighborDiscoveryProtocol):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        mode = kwargs['mode']

        self.slot_duration = kwargs['slot_duration']

        # two "globally fixed" values used by BLT and BL modes
        if 'pit' in kwargs and 'pil' in kwargs:
            self.pit = kwargs['pit']
            self.pil = kwargs['pil']

        # estimated node density or "clique size", if given
        if 'Ne' in kwargs:
            self.Ne = kwargs['Ne']
            self.Nep = self.Ne

        self.set_mode(mode)

        # compute Lambda if P is given using eqn 14 which provides slot count that gives target P
        if 'P' in kwargs:
            # from eqn 14
            self.P = kwargs['P']
            self.compute_Lambda(self.P, self.Ne)

        if 'w' in kwargs:
            self.w = kwargs['w']

        self.current_epoch = 0
        self.Q = I_IDLE
    
    def set_mode(self, mode):
        self.mode = mode

        if mode == 'BLT':  # birthday-listen-and-transmit
            self.pt = self.pit
            self.pl = self.pil
            self.ps = 1 - self.pit - self.pil

        elif mode == 'BL':   # birthday-listen
            self.pt = 0
            self.pl = self.pit
            self.ps = 1 - self.pil

        elif mode == 'PRR':   # probabilistic round robin, tuned to estimated node density
            self.compute_PRR_schedule(self.Ne)

    def compute_Lambda(self, P, Ne):
        if self.mode == 'BL':
            self.Lambda = math.floor(math.log(1 - P)/math.log(1 - (self.pl/self.Ne)))
        elif self.mode == 'PRR':
            self.Lambda = math.floor(-math.log(1 - P)/(self.pt*self.pl**(self.Ne - 1)))

        logging.debug(f'[Birthday] n = {self.Lambda} = Lambda slots to achieve target P = {self.P}')

    def compute_PRR_schedule(self, Ne):
        self.pt = 1/Ne
        self.pl = 1 - (1/Ne)
        self.ps = 0

        logging.debug(f'[Birthday:PRR] pt = {self.pt}, pl = {self.pl}, ps = {self.ps}')

    def schedule(self):
        # random offset first
        offset = self.parent.scene.timeout_event(None, self.parent.start_offset)
        yield offset

        while self.parent.is_alive:
            state = random.choices(['T', 'L', 'S'], k=1, weights=[self.pt, self.pl, self.ps])[0]

            # transmit
            if state == 'T':
                self.Q = I_ADV

                for j in range(0, (self.slot_duration)):
                    logging.debug('[Birthday] transmit')
                    ## create and emit a beaconevent that contains the node's current context
                    bevent = BeaconEvent(self.parent.name, self.parent.position, self.parent.c, 0)
                    beacon = self.parent.scene.timeout_event(bevent, 1)

                    yield beacon
                    logging.debug(f'[Birthday] {self.parent.name} beacon {self.parent.scene.time}')
                    self.parent.scene.queue_event(beacon)
                    self.append_schedule_plot(bevent)

                    self.current_epoch += 1
            # listen
            elif state == 'L':
                self.Q = I_SCAN

                for j in range(0, (self.slot_duration)):
                    logging.debug('[Birthday] listen')
                    sevent = ScanEvent(self.parent.name, self.parent.position)
                    scan = self.parent.scene.timeout_event(sevent, 1)

                    yield scan
                    logging.debug(f'[Birthday] {self.parent.name} scan {self.parent.scene.time}')
                    self.parent.scene.queue_event(scan)
                    self.append_schedule_plot(sevent)

                    self.current_epoch += 1
            # sleep
            elif state =='S':
                self.Q = I_IDLE

                logging.debug('[Birthday] sleep')
                aevent = AdvertisementIntervalEvent(self.parent.name, self.parent.position)
                interval = self.parent.scene.timeout_event(aevent, self.slot_duration)

                self.append_schedule_plot(aevent)
                self.parent.scene.queue_event(interval)
                
                yield interval
                self.append_schedule_plot(aevent)
                self.parent.scene.queue_event(interval)

                self.current_epoch += 1

        logging.info('[Birthday] Schedule terminating.')

    def Fd(self, N, w):
        return 1 - math.exp((-1 * (w/N)) * ((1 - 1/N) ** (N - 1)))

    def D(self, N, w):
        return (self.Fd(N, w) * N)

    def Ndiff(self, Na_guess):
        return (self.D(Na_guess, 2*self.Lambda)) - (self.D(Na_guess, self.Lambda))

    def Ndiff2(self, Na_guess):
        return (self.Eh(Na_guess))
    
    # expected number of links discovered in a single slot
    def Eh(self, Na_guess):
        return Na_guess * (Na_guess - 1) * self.pt * self.pl * ((1 - self.pt) ** (Na_guess - 2))

    """ 
    Ndiff stuff 
    """
    def estimate_Na_from_Ndiffa(self, Ndiffa):
        """ Estimates the actual number of nodes Na using a comparison between the observed Ndiffa
        and the computed Ndiffe for the current schedule.
        """
        Na_guess = 0
        Ndiff = 0

        threshold = 1

        while abs(Ndiffa - Ndiff) > threshold:
            Na_guess += 5
            Ndiff = self.Ndiff(Na_guess)
            logging.info(f'[Birthday] Ndiffa = {Ndiffa}, Ndiff = {Ndiff}, Na_guess = {Na_guess}')

        return Na_guess

    def adapt_schedule(self, Ne):
        logging.info(f'[Birthday] adapting schedule to {Ne}')
        self.compute_PRR_schedule(Ne)       # compute a new schedule
        self.compute_Lambda(self.P, Ne)     # compute a new latency bound for this schedule and node estimate
        self.Nep = Ne
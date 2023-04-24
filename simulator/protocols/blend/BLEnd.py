import random
import logging

from Events import *

from protocols.NeighborDiscoveryProtocol import NeighborDiscoveryProtocol
from protocols.blend.BLEndParameters import BLEndParameters
import protocols.blend.ndiffer as ndiffer

class BLEnd(NeighborDiscoveryProtocol):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.parameters = BLEndParameters.from_optimizer(
            P = kwargs['parameters']['P'],
            Lambda = kwargs['parameters']['Lambda'],
            N = kwargs['parameters']['Ne'],
            beacon_duration = 3,    # these are hardcoded. whatever
            max_random_slack = 10
        )

        # set beacon duration divided by 3 to match what the optimizer outputs
        self.parameters.beacon_duration = 1
        
        # FIXME this is only here so the evaluator works more generally with all protocols. clean it up someday
        self.Ne = self.parameters.N
        self.Nep = self.Ne
        
        self.P = self.parameters.P
        self.Lambda = self.parameters.Lambda

        # logging values used by different evaluators
        self.schedule_plot = []
        self.current_epoch = 0

        self.adapt_next_epoch = False

        # self.Ndiffs = [ndiffer.Ndiff(N, self.parameters.as_dict()) for N in range(5, 105, 5)]
    
    def beacon(self):
        for j in range(0, (self.parameters.beacon_duration)):
            ## create and emit a beaconevent, which contains the node's current context
            bevent = BeaconEvent(self.parent.name, self.parent.position, self.parent.c, j)
            beacon = self.parent.scene.timeout_event(bevent, 1)

            yield beacon
            logging.debug(f'{self.parent.name} beacon {self.parent.scene.time}')
            self.parent.scene.queue_event(beacon)
            self.append_schedule_plot(bevent)

    def ad_interval(self, duration):
        aevent = AdvertisementIntervalEvent(self.parent.name, self.parent.position)
        interval = self.parent.scene.timeout_event(aevent, duration)

        self.append_schedule_plot(aevent)
        self.parent.scene.queue_event(interval)

        # yield for duration of advertising interval
        yield interval
        self.append_schedule_plot(aevent)
        self.parent.scene.queue_event(interval)

    def schedule(self):
        self.current_epoch = 0

        # initalizes a list to track the state of blend at each time t an event is raised
        self.clear_schedule_plot()

        # yield for the given offset before starting the schedule
        # when everyone starts at the same time, it takes a long time for them to start discovering each other
        # it's also not very realistic. so use an offset :)
        offset = self.parent.scene.timeout_event(None, self.parent.start_offset)
        yield offset

        while self.parent.is_alive:
            # the protocol has instructions to adapt to a new node density (i.e., the flag was toggled and there's a new estimated node density)
            if self.adapt_next_epoch:
                    self.parameters = BLEndParameters.from_optimizer(
                        P = self.P,
                        Lambda = self.Lambda,
                        N = self.Nep,
                        beacon_duration = 3,    # these are hardcoded. whatever
                        max_random_slack = 10
                    )

                    self.parameters.beacon_duration = 1
                    # self.Ndiffs = [ndiffer.Ndiff(N, self.parameters.as_dict()) for N in range(2, 100)]

                    self.adapt_next_epoch = False

            epoch_start = self.parent.scene.time

            ## SCAN
            for i in range(0, self.parameters.scan_interval):
                sevent = ScanEvent(self.parent.name, self.parent.position)
                scan = self.parent.scene.timeout_event(sevent, 1)

                yield scan
                logging.debug(f'{self.parent.name} scan {self.parent.scene.time}')
                self.append_schedule_plot(sevent)
                self.parent.scene.queue_event(scan)

            # did the scan complete in the assigned scan interval length?
            assert self.parent.scene.time - epoch_start == self.parameters.scan_interval

            ## ADVERTISE
            advertise_start = self.parent.scene.time

            advertisement_duration = self.parameters.epoch - self.parameters.scan_interval \
                 - (2 * self.parameters.beacon_duration) - self.parameters.advertising_interval \
                - self.parameters.max_random_slack

            # top level loop, one run of inner loop per beacon
            while self.parent.scene.time - advertise_start < advertisement_duration:
                beacon_start = self.parent.scene.time
            
                # inner loop for the beacon itself, which is actually divided into multiple sequential beacon events
                yield from self.beacon()

                # did the beacon complete in the assigned beacon length?
                assert self.parent.scene.time - beacon_start == (self.parameters.beacon_duration)

                yield from self.ad_interval(self.parameters.advertising_interval + random.randint(0, self.parameters.max_random_slack))

            # FIXME figure out what the last advertising interval should actually look like and implement it here.
            ## last advertisement block, which is done without (?) slack (or with slack determined by the amount of wiggle room left?)
            yield from self.beacon()

            last_interval_duration = self.parameters.epoch - (self.parent.scene.time - epoch_start) \
                - self.parameters.beacon_duration

            if last_interval_duration > 0:
                yield from self.ad_interval(last_interval_duration)
                yield from self.beacon()
            # there is not enough time remaining for another interval + beacon, so just delay for a few ms to pad it out
            else:
                yield from self.ad_interval(self.parameters.epoch - (self.parent.scene.time - epoch_start))

            # did the epoch complete in the assigned epoch length?
            assert self.parent.scene.time - epoch_start == self.parameters.epoch, f'{self.parent.scene.time - epoch_start} != {self.parameters.epoch}; {last_interval_duration}'

            self.current_epoch += 1
            self.parameters.update_energy_consumption()

        logging.info('[BLEnd] Schedule terminating.')

        ## this may not be ideal when using real data, since we are more interested in replaying all of the data available
        self.parent.is_alive = False

    def estimate_Na_from_Ndiffa(self, Ndiffa):
        diff = 9999 # this feels hacky
        closest = self.Nep

        guesses = list(range(5, 105, 5))
        guesses.insert(0, 2)

        for guess in guesses:
            Ndiff = ndiffer.Ndiff(guess, self.parameters.as_dict())
            
            if abs(Ndiff - Ndiffa) < diff:
                diff = abs(Ndiff - Ndiffa)
                closest = guess

        logging.info(f'[BLEnd] Ndiffa = {Ndiffa}, closest Ndiff = {Ndiff}, difference = {diff}')

        return closest

    """ Doesn't adapt the schedule immediately – waits till the start of the next epoch. """
    def adapt_schedule(self, Ne):
        logging.info(f'[BLEnd] adapting schedule to {Ne} on next epoch')
        self.adapt_next_epoch = True
        self.Nep = Ne
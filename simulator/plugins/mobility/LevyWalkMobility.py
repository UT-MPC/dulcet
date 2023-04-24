import pickle
import numpy as np
import logging

from os import path

from scipy.stats import uniform
from scipy.stats import levy

from plugins.mobility.MobilityPlugin import MobilityPlugin
import plugins.mobility.LevyWalkModel as lw

THRES_LOC = 70
THRES_TIME = 250
ALPHA = 0.8
BETA = 1.5
TIME_STEP_SIZE = 0.001

class LevyWalkMobility(MobilityPlugin):
    def __init__(self, **kwargs):
        self.parent_node = kwargs['parent_node']

        bounds = kwargs['bounds']
        
        if 'static_region' in kwargs:
            static_region = kwargs['static_region']
        else:
            static_region = None

        # default to ms if no units given
        try:
            stop_time = int(kwargs['stop_time'])
        except ValueError:
            stop_time = kwargs['stop_time'].split()
            units = stop_time[1]
            stop_time = int(stop_time[0])

            if units == 'sec':
                stop_time *= 1000
            elif units == 'min':
                stop_time *= 60000
            elif units == 'hr':
                stop_time *= 3600000
            else:
                logging.warning('[LevyWalkMobility] Unrecognized units given for stop time.')
                sys.exit()

        # check to see if there is a pickle of the mobility trace already, rather than regenerating it every time
        cur = path.dirname(path.dirname(path.abspath(__file__)))
        fn = path.join(cur, 'cache', 'mobility', f'{kwargs["parent_node"]}_{kwargs["bounds"]}_{kwargs["steps"]}_{kwargs["episodes"]}_{stop_time}.pkl')

        if path.exists(fn):
            logging.debug(f'[LevyWalkMobility][{kwargs["parent_node"]}] loading mobility trace pickle')
            
            with open(fn, 'rb') as f:
                self.trace = pickle.load(f)

            self.position = (self.trace[0][0], self.trace[0][1])

        else:
            x0 = (np.random.rand(1)*(2 * bounds[0]) + -bounds[0])[0]
            y0 = (np.random.rand(1)*(2 * bounds[1]) + -bounds[1])[0]

            self.position = (x0, y0)

            positions, times = lw.levy_walk_episodes(
                kwargs['steps'], 
                kwargs['bounds'], 
                self.position[0], 
                self.position[1],
                THRES_LOC, 
                THRES_TIME, 
                TIME_STEP_SIZE,
                ALPHA, 
                BETA,
                kwargs['episodes'],
                static_region
            )

            positions = positions.tolist()
            times = times.tolist()

            # rescale time domain depending on the desired stop time
            t_min = min(times)
            t_max = max(times)
            times = [int(((t - t_min)/(t_max - t_min))*(stop_time - t_min)+t_min) for t in times]

            self.trace = dict(zip(times, positions))

            # save a pickle of the mobility trace so we don't have to generate it again
            with open(fn, 'wb') as f:
                pickle.dump(self.trace, f)

    def update(self, parameters):
        if parameters in self.trace:
            self.position = self.trace[parameters]
            logging.debug(f'[LevyWalkMobility][{self.parent_node}] position: {self.position}')
        
        return self.position
# implements a Scene, which wraps the simulator logic with logic for processing different events at each step
import simpy
from os import path, remove
from math import dist
import logging 
import sys
import pickle

from plugins.evaluator.EvaluatorPlugin import *

from plugins.mobility.MobilityPlugin import StaticMobility
from plugins.context.ContextPlugin import StaticContext

from protocols.blend.BLEnd import BLEnd
from protocols.birthday.Birthday import Birthday

from Events import *
from RuntimeEstimator import RuntimeEstimator

# how many events to batch before dumping a pickle of the simulation event queue
PICKLE_BATCH_SIZE = 100

class Scene:
    """ A Scene, which wraps the passed simulation environment in additional logic 
    for processing events at each time step.
    """

    def __init__(self, stop_time: int=-1, node_controller={}, communication_range: int=300, pickle_path=None, log_collisions=False):
        # the simpy sim environment
        global env
        env = simpy.Environment()
        self.env = env

        self.communication_range = communication_range

        # stores the batch of events for each time slice
        self.event_batch = []

        # tracks the scene time relative to the sim time - this is how we know when to process the event batch
        self.time = env.now

        # defines the time step to stop the sim at
        self.stop_time = stop_time

        # dict for tracking nodes in the form { j: Node }
        self.nodes = {}

        self.node_indices = {}

        # list of evaluator callables that run different evals at different times
        self.evaluators = {}

        # a top-level controller for the nodes
        self.node_controller = node_controller

        # tracks the number of collisions on each channel at each time step
        self.log_collisions = log_collisions
        self.collisions = []

        self.dead_nodes = []

        self.pickle_path = pickle_path

        # if a pickle path is given, instantiate an empty list for the batch of events to pickle,
        # resolve the full path to the pickle, and wipe any old pickle if it exists
        if pickle_path is not None:
            cur = path.dirname(path.dirname(path.abspath(__file__)))
            self.pickle_path = path.join(cur, 'simulator', 'cache', 'simulations', pickle_path)

            # TODO this will erase the old pickle even when intentionally doing multiple runs of the same sim
            if path.exists(self.pickle_path):
                remove(self.pickle_path)

            # instantiate the pickle batch with the header row
            self.pickle_batch = [['t', 'emitter', 'position', 'event']]
        else:
            self.pickle_batch = None

        # the main loop of the simulator which processes queued events at each 1ms time step
        self.env_process(self.step())

        self.runtime = RuntimeEstimator(stop_time=stop_time)

        self.is_running = True

    def timeout_event(self, event, delay):
        return simpy.events.Timeout(env, delay=delay, value=event)

    def env_timeout(self, delay):
        return env.timeout(delay)

    def env_now(self):
        return env.now

    def env_run(self):
        if self.stop_time != -1:
            env.run(until=self.stop_time+1000)
        else:
            env.run()

    def env_stop(self):
        env.event().succeed()

    def set_evaluators(self, evaluators):
        self.evaluators = evaluators

    def update_node_neighbors(self):
        nodes = list(self.nodes.items())

        for i, node1 in enumerate(nodes):
            node1 = node1[1]
            for node2 in nodes[i+1:]:
                node2 = node2[1]
                if dist(node1.position, node2.position) <= self.communication_range:
                    node1.add_G_neighbor(self.time, node2.name, node2.c)
                    node2.add_G_neighbor(self.time, node1.name, node1.c)
                else:
                    node1.del_G_neighbor(node2.name)
                    node2.del_G_neighbor(node1.name)

    def update_single_node_neighbors(self, node):
        nodes = list(self.nodes.values())

        for i, neighbor in enumerate(nodes):
            if dist(node.position, neighbor.position) <= self.communication_range and node != neighbor:
                node.add_G_neighbor(self.time, neighbor.name, neighbor.c)
                neighbor.add_G_neighbor(self.time, node.name, node.c)
            else:
                i = self.node_indices[neighbor.name]
                j = self.node_indices[node.name]

                self.adjacency_matrix[i][j] = 0
                self.adjacency_matrix[j][i] = 0

                node.del_G_neighbor(neighbor.name)
                neighbor.del_G_neighbor(node.name)

    def log_collision(self, channel, num_colliders):
        if self.log_collisions:
            self.collisions.append((self.time, channel, num_colliders))

    def update_from_node_controller(self):
        # create new nodes according to the node controller
        if str(self.time) in self.node_controller:
            if self.node_controller[str(self.time)] > 0: # add nodes
                logging.info(f'[Scene][{self.time}] adding {self.node_controller[str(self.time)]} nodes')
                node = list(self.nodes.values())[0]

                # create new blend nodes
                if isinstance(node.ndprotocol, BLEnd):
                    parameters = {
                        'P': node.ndprotocol.P,
                        'Lambda': node.ndprotocol.Lambda,
                        # copy the node's guess, or copy its original estimate...
                        'Ne': node.ndprotocol.Nep
                        #'Ne': node.ndprotocol.Ne
                    }

                    # make the nodes
                    for i in range(0, self.node_controller[str(self.time)]):
                        mobility_parameters = {'position': (0, 0)}

                        Node(self, 
                            f'{random.getrandbits(16)}', 
                            StaticMobility(**mobility_parameters),
                            StaticContext(f'{i}'), 
                            BLEnd(parameters=parameters),
                            is_adaptive=node.is_adaptive,
                            w=node.w,
                            w_mult=node.w_mult,
                            si_samples=node.si_samples,
                            adaptation_start_offset=self.time
                        )

                # or create new birthday nodes
                elif isinstance(node.ndprotocol, Birthday):
                    parameters = {
                        'P': node.ndprotocol.P,
                        'slot_duration': 1,
                        'mode': 'PRR',
                        # copy the node's guess, or copy its original estimate...
                        'Ne': node.ndprotocol.Nep
                        #'Ne': node.ndprotocol.Ne
                    }

                    # make the nodes
                    for i in range(0, self.node_controller[str(self.time)]):
                        mobility_parameters = {'position': (0, 0)}

                        Node(self, 
                            f'{random.getrandbits(16)}', 
                            StaticMobility(**mobility_parameters),
                            StaticContext(f'{i}'), 
                            Birthday(**parameters),
                            is_adaptive=node.is_adaptive,
                            w=node.w,
                            w_mult=node.w_mult,
                            si_samples=node.si_samples,
                            adaptation_start_offset=self.time
                        )
            
            else: # remove nodes
                count = -self.node_controller[str(self.time)]
                logging.info(f'[Scene][{self.time}] removing {count} nodes')
                
                goners = list(self.nodes.items())[0:count]

                for k, node in goners:
                    self.dead_nodes.append(k)
                    node.is_alive = False
                    del self.nodes[k]

                # TODO could optimize this
                # look at every node
                for j, node in self.nodes.items():
                    # look at every node's discoveries
                    for k, v in list(node.C.items()):
                        # if that discovery came from a deleted node, delete the discovery
                        if k in list(zip(*goners))[0]:
                            if k in node.C:
                                del node.C[k]
                    # look at every node's ground truth neighbors
                    for k, v in list(node.G.items()):
                        # if that neighbor has been deleted, remove it from the ground truth set
                        if k in list(zip(*goners))[0]:
                            if k in node.G:
                                del node.G[k]
                
                logging.info(f'[Scene][{self.time}] Na = {len(self.nodes)}')

    def update_pickle(self, event):
        # if we're pickling the sim and the current batch is at the desired batch size, dump it
        if self.pickle_batch is not None:
            if len(self.pickle_batch) <= PICKLE_BATCH_SIZE:
                self.pickle_batch.append([self.time, event.origin_id, event.origin_position, type(event)])
            elif len(self.pickle_batch) >= PICKLE_BATCH_SIZE:
                with open(self.pickle_path, 'ab') as f:
                    pickle.dump(self.pickle_batch, f)
                    self.pickle_batch = []

    def print_progress(self, time_left=''):
        if logging.root.level <= logging.INFO:
            progress = f'[Scene] {self.env_now()}/{self.stop_time} ({int((self.env_now()/self.stop_time)*100)}%) {time_left}'
            print(progress + ' ' * len(progress) + '\r', end='')

    def step(self):
        self.update_node_neighbors()
        self.adjacency_matrix = self.get_discovery_adjacency_matrix()

        while True:
            self.runtime.start_block(self.env_now())

            self.update_from_node_controller()

            # we're on the next time slice. time to process all of the events from the previous time slice
            ## this is where discovery is implemented. for each ScanEvent, determine if there was a corresponding
            ## BeaconEvent, which implies discovery could occur
            if self.env_now() == self.time + 1:
                logging.debug(f'[Scene][{self.time}] processing batch')
                
                # for every scan event, we want to check which (if any) beacon events were discovered
                for event in self.event_batch:
                    # update the node's neighbors if the event is relevant to their position
                    if (isinstance(event, ScanEvent) or isinstance(event, BeaconEvent)) and event.origin_id in self.nodes:
                        self.update_single_node_neighbors(self.nodes[event.origin_id])

                    self.update_pickle(event)

                    if isinstance(event, ScanEvent):
                        scanner = (event.origin_id, event.origin_position)
                        beacons = [event for event in self.event_batch if isinstance(event, BeaconEvent)]

                        logging.debug(f'[Scene] {scanner} process scan {self.time}')

                        if beacons:
                            logging.debug(f'[Scene] {len(beacons)} beacon(s) discoverable by scanner {scanner[0]}')

                            # track collisions in each channel
                            colliders = {
                                0: [],
                                1: [],
                                2: []
                            }

                            ## process the beacons
                            for beacon in beacons:
                                advertiser = (beacon.origin_id, beacon.origin_position)

                                if self.discovery_probability(scanner, advertiser) == 1 and scanner[0] != advertiser[0]:
                                    colliders[beacon.channel].append(beacon)

                                logging.debug(f'[Scene] {advertiser} process beacon {self.time} channel {beacon.channel}')

                            for channel in colliders.keys():
                                # track the number of collisions on this step in a tuple (t, channel, collisions)
                                self.log_collision(channel, len(colliders[channel])-1)

                                ## if there are colliders on this channel, no discovery occurs
                                ## if there is only one beacon arriving, discovery occurs
                                if len(colliders[channel]) == 1:
                                    beacon = colliders[channel].pop()
                                    
                                    # update the scanner if they're still around. when doing more advanced stuff with nodes (adding and removing
                                    # them over time), sometimes scanners will disappear from the sim before their scan events can be processed
                                    if scanner[0] in self.nodes and beacon.origin_id not in self.dead_nodes:
                                        logging.debug(f'[Scene] discovery')
                                        self.nodes[scanner[0]].update_C_neighbor(self.time, beacon.origin_id, beacon.payload)
                                        
                                        # update adjacency matrix
                                        i = self.node_indices[scanner[0]]
                                        j = self.node_indices[beacon.origin_id]

                                        self.adjacency_matrix[i][j] = 1

                                        # add a discovery event to the pickle if we're making one
                                        # this is so we can measure radio time spent "receiving" later on (if we feel like it)
                                        if self.pickle_path is not None:
                                            devent = DiscoveryEvent(scanner[0], scanner[1])
                                            self.pickle_batch.append([self.time, devent.origin_id, devent.origin_position, type(devent)])
                                            self.nodes[scanner[0]].ndprotocol.append_schedule_plot(devent)

                                    elif beacon.origin_id in self.dead_nodes:
                                        logging.debug(f'[Scene] ghost beacon')
                                    else:
                                        logging.debug(f'[Scene] {len(colliders[channel])} collisions (channel {channel})')

                self.time = env.now
                self.event_batch = []
            
            # update evaluators
            for evaluator in self.evaluators:
                evaluator.update()

            self.print_progress(time_left=self.runtime.end_block())
            
            yield self.env_timeout(1)

    def env_process(self, f):
        env.process(f)

    def discovery_probability(self, scanner, advertiser):
        distance = dist(scanner[1], advertiser[1])

        ## determine if the beacon is in range
        ## TODO this could be fancy. ex: determine the range as a function of tx_power.
        if distance <= self.communication_range:
            return 1.00
        
        return 0.00

    def queue_event(self, event):
        """ Batches events for a time slice, processing them when the next time slice is reached
        """
        if event.value:
            logging.debug(f'{event.value.origin_id} {self.time} batch {event.value}')
        self.event_batch.append(event.value)

    def add_node(self, node):
        self.node_indices[node.name] = len(self.nodes)
        self.nodes[node.name] = node        

    def get_discovery_adjacency_matrix(self):
        """ Generates and returns a 2D adjacency matrix representing (directionally) which nodes
        have discovered one another at the current time step (when invoked).
        """
        mat = np.zeros(shape=(len(self.nodes), len(self.nodes)))

        # for every node, get its discovered neighbors
        for name, node in self.nodes.items():
            neighbors = node.C_neighbors()

            # for every discovered neighbor, place a 1 in the matrix
            for neighbor in neighbors:
                mat[self.node_indices[name]][self.node_indices[neighbor]] = 1

        np.fill_diagonal(mat, 0)

        return mat            

from Node import Node
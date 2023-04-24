import logging

import plugins.mobility.MobilityPlugin as MobilityPlugin
import plugins.context.ContextPlugin as ContextPlugin
from protocols import NeighborDiscoveryProtocol
from protocols.adaptive import AdaptationProtocol
from Events import *
from Scene import Scene

class Node:
    """ An individual simulation Node.
    
    Nodes possess a position, a context, and a set of parameters that define the duty cycle of 
    continuous neighbor discovery. The Node's position and context are mutated on each update 
    by the MobilityPlugin and ContextPlugin (respectively) passed to the Node's constructor.
    """
    def __init__(self, scene: Scene, name: str, mobility: MobilityPlugin, context: ContextPlugin, 
        ndprotocol: NeighborDiscoveryProtocol, adaptation_protocol: AdaptationProtocol=None, start_offset: int=0):
        
        self.is_alive = False

        self.scene = scene
        self.name = name
        self.mobility = mobility
        self.context = context

        # stores a reference to the neighbor discovery protocol
        self.ndprotocol = ndprotocol
        self.ndprotocol.set_parent(self)

        self.adaptation_protocol = adaptation_protocol
        if adaptation_protocol is not None:
            self.adaptation_protocol.set_parent(self)

        # an offset to wait before starting blend - used so that all nodes don't start at the same time
        self.start_offset = start_offset

        # stores known neighbors with a timestamp indiciating when they were discovered
        ## old neighbors are periodically discarded according to tau if we aren't receiving their beacons anymore
        ## {'name': (discovery_time, payload)}
        self.C = {}

        # stores ground truth neighbors - other nodes in range, regardless of whether this node has discovered them
        self.G = {}

        # do an update step to initialize context and position
        self.update()

        # add this node's cnd schedule to the simulation environment
        self.scene.env_process(self.ndprotocol.schedule())

        # add this node's state update schedule
        self.scene.env_process(self.step())

        self.is_alive = True

        # explicitly add the node to the scene - although it technically exists to the sim by this point,
        # the scene needs to know about it too in order to detect neighbors
        self.scene.add_node(self)

    def update(self):
        logging.debug(f'[Node:{self.name}][{self.scene.env_now()}] update')

        # update node simulation state (position and context value)
        self.position = self.mobility.update(self.scene.time)
        self.c = self.context.update(data=self.C)

    def step(self):
        while self.is_alive:
            assert self.name not in self.scene.dead_nodes, 'Node is dead but still running!'
            
            self.update()
            yield self.scene.env_timeout(1)
            
            if self.adaptation_protocol is not None:
                self.adaptation_protocol.adapt_schedule()

    def purge_neighbors(self):
        purged_neighbors = []

        ## update neighbors: purge those that we haven't heard from in a while
        for j, d in self.C.items():
            if self.scene.env_now() - d[0] > self.tau:
                logging.debug(f'[Node:{self.name}][{self.scene.env_now()}] purging {j} from C')
                purged_neighbors.append(j)

        for j in purged_neighbors:
            del self.C[j]

    def C_completeness(self):
        """ Computes the completeness of C at the current time step.

        Completeness measures what percent of unique d in G are in C, i.e. how many
        true neighbors this Node is currently aware of.
        """
        if len(self.G.keys()) == 0:
            return float('NaN')
        
        known = 0

        for j in self.C.keys():
            if j in self.G.keys():
                known += 1

        return known/len(self.G.keys())

    def C_cardinality(self):
        return len(self.C.keys())

    def C_neighbors(self):
        return self.C.keys()

    def clear_C(self):
        self.C = {}

    def has_C_neighbor(self, j):
        return j in self.C

    def update_C_neighbor(self, t, j, c):
        logging.debug(f'[Node:{self.name}][{self.scene.env_now()}] new C neighbor {j}')
        self.C[j] = (t, c)

        if self.adaptation_protocol is not None:
            self.adaptation_protocol.update_U_neighbor(t, j, c)

    def has_G_neighbor(self, j):
        return j in self.G

    def add_G_neighbor(self, t, j, c):
        self.G[j] = (t, c)

    def del_G_neighbor(self, j):
        if j in self.G:
            del self.G[j]
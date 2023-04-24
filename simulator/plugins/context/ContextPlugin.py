# defines some internal logic that determines how the context of a node is computed at each step
# basically, this is an abstraction of some context value that will change over time - maybe a sensor
# reading, or some other value which will be computed internally to the node and then shared with others
# in the broadcast beacons
from abc import ABC, abstractmethod

class ContextPlugin(ABC):
    @abstractmethod
    def update(self, data=None):
        pass

class StaticContext(ContextPlugin):
    def __init__(self, context):
        self.context = context

    def update(self, data=None):
        return self.context
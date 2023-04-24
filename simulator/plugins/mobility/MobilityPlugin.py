from abc import ABC, abstractmethod

class MobilityPlugin(ABC):
    @abstractmethod
    def update(self, parameters):
        pass

class StaticMobility(MobilityPlugin):
    def __init__(self, **kwargs):
        self.position = kwargs['position']

    def update(self, parameters):
        return self.position
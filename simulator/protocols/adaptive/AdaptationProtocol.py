# defines an interface for a neighbor discovery protocol
from abc import ABC, abstractmethod

class AdaptationProtocol(ABC):
    def __init__(self, **kwargs):
        pass

    def set_parent(self, parent):
        self.parent = parent
    
    @abstractmethod
    def adapt_schedule(self):
        pass
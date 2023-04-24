# defines an interface for a neighbor discovery protocol
from abc import ABC, abstractmethod

class NeighborDiscoveryProtocol(ABC):
    def __init__(self, **kwargs):
        self.schedule_plot = []

        if 'log_schedule' in kwargs:
            self.log_schedule = kwargs['log_schedule']
        else:
            self.log_schedule = False

    def set_parent(self, parent):
        self.parent = parent

    def clear_schedule_plot(self):
        self.schedule_plot = []

    def append_schedule_plot(self, event):
        if self.log_schedule:
            self.schedule_plot.append((self.parent.scene.time, int(event)))

    @abstractmethod
    def schedule():
        pass

    @abstractmethod
    def estimate_Na_from_Ndiffa(self, Ndiffa):
        pass

    @abstractmethod
    def adapt_schedule(self, Ne):
        pass
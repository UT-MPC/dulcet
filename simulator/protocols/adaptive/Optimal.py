import logging

from protocols.adaptive.AdaptationProtocol import AdaptationProtocol

class Optimal(AdaptationProtocol):
    def __init__(self, **kwargs):
        pass

    def adapt_schedule(self):
        Na = len(self.parent.G) + 1
        
        if Na != self.parent.ndprotocol.Nep and Na > 0:
            self.parent.ndprotocol.adapt_schedule(Na)

    def update_U_neighbor(self, t, j, c):
        pass
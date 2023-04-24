
class BeaconEvent:
    """ Each BeaconEvent contains:
        - a reference to the node that advertised it
        - the payload, which is an activity classification for the node that sent it
    """
    def __init__(self, origin_id: str, origin_position: tuple, payload, channel: int):
        self.origin_id = origin_id
        self.origin_position = origin_position
        self.payload = payload
        self.channel = channel

    def __int__(self):
        return 1

class ScanEvent:
    """ Emitted by a NeighborDiscoveryProtocol when it is scanning for neighbors. 
    """
    def __init__(self, origin_id: str, origin_position: tuple):
        self.origin_id = origin_id
        self.origin_position = origin_position

    def __int__(self):
        return 2

class DiscoveryEvent:
    """ Emitted by the Scene when two Nodes discover one another.
    """
    def __init__(self, origin_id: str, origin_position: tuple):
        self.origin_id = origin_id
        self.origin_position = origin_position

    def __int__(self):
        return 0

class AdvertisementIntervalEvent:
    """ Emitted by a NeighborDiscoveryProtocol during its advertising intervals between beacons.
    """
    def __init__(self, origin_id: str, origin_position: tuple):
        self.origin_id = origin_id
        self.origin_position = origin_position

    def __int__(self):
        return -1
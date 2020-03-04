from collections import deque

from ophyd.sim import NullStatus, new_uid


class BlueskyFlyer:
    def __init__(self):
        self.name = 'bluesky_flyer'
        self._asset_docs_cache = deque()
        self._resource_uids = []
        self._datum_counter = None
        self._datum_ids = []

    def kickoff(self):
        return NullStatus()

    def complete(self):
        return NullStatus()

    def collect(self):
        ...

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item


class HardwareFlyer(BlueskyFlyer):
    def __init__(self, params_to_change, detector, motors):
        super().__init__()
        self.params_to_change = params_to_change
        self.detector = detector
        self.motors = motors

    def kickoff(self):
        # get initial positions of each motor
        # calculate distances to travel
        # calculate velocities
        # start movement
        # motors status returned, use later in complete
        return NullStatus()

    def complete(self):
        # all motors arrived
        return NullStatus()

    def collect(self):
        ...

params_to_change = []

motors = {sample_stage.x.name: sample_stage.x,
          sample_stage.y.name: sample_stage.y,
          sample_stage.z.name: sample_stage.z,}

params_to_change.append({sample_stage.x.name: -1,
                         sample_stage.y.name: 0,
                         sample_stage.z.name: 1})

params_to_change.append({sample_stage.x.name: 1,
                         sample_stage.y.name: -1,
                         sample_stage.z.name: 0})

params_to_change.append({sample_stage.x.name: 0,
                         sample_stage.y.name: 1,
                         sample_stage.z.name: -1})

# update function - change params
hf = HardwareFlyer(params_to_change=params_to_change, detector=xs, motors=motors)
# RE(bp.fly([hf]))

def optimize():
    # do stuff
    yield from bp.fly()

from collections import deque

from ophyd.sim import NullStatus, new_uid

import numpy as np


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

def calc_velocity(motors, dists, vels):
    ret_vels = []
    # find max distance to move
    max_dist = np.max(dists)
    max_dist_index = dists.index(max_dist)
    max_dist_vel = vels[max_dist_index][1]
    time_needed = dists[max_dist_index] / max_dist_vel
    for i in range(len(vels)):
        if i != max_dist_index:
            try_vel = np.round(dists[i] / time_needed, 1)
            if try_vel < vels[i][0]:
                try_vel = vels[i][0]
            elif try_vel > vels[i][1]:
                break
            else:
                ret_vels.append(try_vel)
        else:
            ret_vels.append(vels[max_dist_index][1])
    if len(ret_vels) == len(motors):
        # if all velocities work, return
        return ret_vels
    else:
        # try using slowest motor to calculate time
        ret_vels.clear()
        # find slowest motor
        slow_motor_index = np.argmin(vels, axis=0)[1]
        slow_motor_vel = vels[slow_motor_index][1]
        time_needed = dists[slow_motor_index] / slow_motor_vel
        for j in range(len(vels)):
            if j != slow_motor_index:
                try_vel = np.round(dists[i] / time_needed, 1)
                if try_vel < vels[i][0]:
                    try_vel = vels[i][0]
                elif try_vel > vels[i][1]:
                    break
                else:
                    ret_vels.append(try_vel)
            else:
                ret_vels.append(vels[slow_motor_index][1])
        return ret_vels

def optimize():
    # do stuff
    yield from bp.fly()

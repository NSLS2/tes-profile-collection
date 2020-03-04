import time as ttime
import bluesky.plans as bp
import bluesky.plan_stubs as bps

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

    def describe_collect(self):
        return {self.name: {}}

    def collect(self):
        now = ttime.time()
        data = {}
        yield {'data': data,
               'timestamps': {key: now for key in data},
               'time': now,
               'filled': {key: False for key in data}}

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item


class HardwareFlyer(BlueskyFlyer):
    def __init__(self, params_to_change, velocities, time_to_travel, detector, motors):
        super().__init__()
        self.name = 'tes_hardware_flyer'

        # TODO: These 3 lists to be merged later
        self.params_to_change = params_to_change  # dictionary with motor names as keys
        self.velocities = velocities  # dictionary with motor names as keys
        self.time_to_travel = time_to_travel  # dictionary with motor names as keys

        self.detector = detector
        self.motors = motors

        self.watch_positions = {name: [] for name in self.motors}
        self.watch_intensities = []
        self.watch_timestamps = []

        self.motor_move_status = None

    def kickoff(self):
        # get initial positions of each motor (done externally)
        # calculate distances to travel (done externally)
        # calculate velocities (done externally)
        # preset the velocities (done in the class)
        # start movement (done in the class)
        # motors status returned, use later in complete (done in the class)

        slowest_motor = sorted(self.time_to_travel,
                               key=lambda x: self.time_to_travel[x],
                               reverse=True)[0]

        self.detector.settings.acquire.put(0)
        self.detector.settings.num_images.put(10000)
        self.detector.settings.acquire_time.put(0.05)
        self.detector.settings.acquire.put(1)

        # Call this function once before we start moving all motors to collect the first points.
        self._watch_function()

        # TODO: add a check for zero division:
        # ~/.ipython/profile_collection/startup/92-optimization.py in calc_velocity(motors, dists, velocity_limits)
        #     169     for i in range(len(velocity_limits)):
        #     170         if i != max_dist_index:
        # --> 171             try_vel = np.round(dists[i] / time_needed, 1)
        #     172             if try_vel < velocity_limits[i][0]:
        #     173                 try_vel = velocity_limits[i][0]
        #
        # ZeroDivisionError: float division by zero
        for motor_name, motor_obj in self.motors.items():
            motor_obj.velocity.put(self.velocities[motor_name] / 20)

        for motor_name, motor_obj in self.motors.items():
            if motor_name == slowest_motor:
                self.motor_move_status = motor_obj.set(self.params_to_change[motor_name])
            else:
                motor_obj.set(self.params_to_change[motor_name])

        self.motor_move_status.watch(self._watch_function)

        return NullStatus()

    def complete(self):
        # all motors arrived
        return self.motor_move_status

    def describe_collect(self):

        return_dict = {self.name:
                           {
                               f'{self.name}_{self.detector.channel1.rois.roi01.name}':
                                   {'source': f'{self.name}_{self.detector.channel1.rois.roi01.name}',
                                    'dtype': 'number',
                                    'shape': []},
                            }
                        }

        motor_dict = {}
        for motor_name, motor_obj in self.motors.items():
             motor_dict[f'{self.name}_{motor_name}_velocity'] = {'source': f'{self.name}_{motor_name}_velocity',
                                                                 'dtype': 'number', 'shape': []}
             motor_dict[f'{self.name}_{motor_name}_position'] = {'source': f'{self.name}_{motor_name}_position',
                                                                 'dtype': 'number', 'shape': []}
        return_dict[self.name].update(motor_dict)

        print('describe_collect:\n', return_dict)

        return return_dict

    def collect(self):
        for ind in range(len(self.watch_intensities)):
            motor_dict = {}
            for motor_name, motor_obj in self.motors.items():
                motor_dict.update(
                    {f'{self.name}_{motor_name}_velocity': self.velocities[motor_name],
                     f'{self.name}_{motor_name}_position': self.watch_positions[motor_name][ind]}
                )

            data = {f'{self.name}_{self.detector.channel1.rois.roi01.name}': self.watch_intensities[ind]}
            data.update(motor_dict)

            print('data:\n', data)

            yield {'data': data,
                   'timestamps': {key: self.watch_timestamps[ind] for key in data},
                   'time': self.watch_timestamps[ind],
                   'filled': {key: False for key in data}}

        ## This will produce one event with dictionaries in the <...>_parameters field.
        # motor_params_dict = {}
        # for motor_name, motor_obj in self.motors.items():
        #     motor_parameters = {'timestamps': self.watch_timestamps,
        #                         'velocity': self.velocities[motor_name],
        #                         'positions': self.watch_positions[motor_name]}
        #     motor_params_dict[motor_name] = motor_parameters
        #
        # data = {f'{self.name}_{self.detector.channel1.rois.roi01.name}': self.watch_intensities,
        #         f'{self.name}_parameters': motor_params_dict}
        #
        # now = ttime.time()
        # yield {'data': data,
        #        'timestamps': {key: now for key in data}, 'time': now,
        #        'filled': {key: False for key in data}}

    def _watch_function(self, *args, **kwargs):
        self.watch_intensities.append(self.detector.channel1.rois.roi01.value.get())
        for motor_name, motor_obj in self.motors.items():
            self.watch_positions[motor_name].append(motor_obj.user_readback.get())
        self.watch_timestamps.append(ttime.time())


params_to_change = []

motors = {sample_stage.x.name: sample_stage.x,
          sample_stage.y.name: sample_stage.y,
          sample_stage.z.name: sample_stage.z,}

"""
# Velocities (same for each motor x, y, z)
In [3]: sample_stage.z.velocity.limits                                                                            
Out[3]: (0.0, 11.0)

# Limits
In [4]: sample_stage.x.limits                                                                                     
Out[4]: (16.0, 88.2)

In [5]: sample_stage.y.limits                                                                                     
Out[5]: (32.0, 116.9)

In [12]: sample_stage.z.limits                                                                                    
Out[12]: (14.0, 23.0)


# Current positions
In [10]: sample_stage.x.user_readback.get()                                                                       
Out[10]: 69.5

In [9]: sample_stage.y.user_readback.get()                                                                        
Out[9]: 40.0

In [11]: sample_stage.z.user_readback.get()                                                                       
Out[11]: 22.22
"""

# TODO: merge "params_to_change" and "velocities" lists of dictionaries to become lists of dicts of dicts.
params_to_change.append({sample_stage.x.name: 72,
                         sample_stage.y.name: 41,
                         sample_stage.z.name: 20})

params_to_change.append({sample_stage.x.name: 68,
                         sample_stage.y.name: 43,
                         sample_stage.z.name: 19})

params_to_change.append({sample_stage.x.name: 71,
                         sample_stage.y.name: 39,
                         sample_stage.z.name: 22.9})

# update function - change params

# RE(bp.fly([hf]))




def calc_velocity(motors, dists, velocity_limits):
    ret_vels = []
    # find max distance to move
    max_dist = np.max(dists)
    max_dist_index = dists.index(max_dist)
    max_dist_vel = velocity_limits[max_dist_index][1]
    time_needed = dists[max_dist_index] / max_dist_vel
    for i in range(len(velocity_limits)):
        if i != max_dist_index:
            try_vel = np.round(dists[i] / time_needed, 1)
            if try_vel < velocity_limits[i][0]:
                try_vel = velocity_limits[i][0]
            elif try_vel > velocity_limits[i][1]:
                break
            else:
                ret_vels.append(try_vel)
        else:
            ret_vels.append(velocity_limits[max_dist_index][1])
    if len(ret_vels) == len(motors):
        # if all velocities work, return
        return ret_vels
    else:
        # try using slowest motor to calculate time
        ret_vels.clear()
        # find slowest motor
        slow_motor_index = np.argmin(velocity_limits, axis=0)[1]
        slow_motor_vel = velocity_limits[slow_motor_index][1]
        time_needed = dists[slow_motor_index] / slow_motor_vel
        for j in range(len(velocity_limits)):
            if j != slow_motor_index:
                try_vel = np.round(dists[i] / time_needed, 1)
                if try_vel < velocity_limits[i][0]:
                    try_vel = velocity_limits[i][0]
                elif try_vel > velocity_limits[i][1]:
                    break
                else:
                    ret_vels.append(try_vel)
            else:
                ret_vels.append(velocity_limits[slow_motor_index][1])
        return ret_vels

hf_flyers = []

def optimize():
    velocities_list = []
    distances_list = []
    for param in params_to_change:
        velocities_dict = {}
        distances_dict = {}
        dists = []
        velocity_limits = []
        for motor_name, motor_obj in motors.items():
            velocity_limits.append(tuple(motor_obj.velocity.limits))
            dists.append(abs(param[motor_name] - motor_obj.user_readback.get()))
        velocities = calc_velocity(motors.keys(), dists, velocity_limits)
        for motor_name, vel, dist in zip(motors, velocities, dists):
            velocities_dict[motor_name] = vel
            distances_dict[motor_name] = dist
        velocities_list.append(velocities_dict)
        distances_list.append(distances_dict)

    # Validation
    times_list = []
    for dist, vel in zip(distances_list, velocities_list):
        times_dict = {}
        for motor_name, motor_obj in motors.items():
            time_ = dist[motor_name] / vel[motor_name]
            times_dict[motor_name] = time_
        times_list.append(times_dict)

    print('Distances:  ', distances_list)
    print('Velocities: ', velocities_list)
    print('Times: ', times_list)

    """
    In [2]: params_to_change                                                                                          
    Out[2]: 
    [{'sample_stage_x': 55, 'sample_stage_y': 60, 'sample_stage_z': 15},
     {'sample_stage_x': 20, 'sample_stage_y': 90, 'sample_stage_z': 18},
     {'sample_stage_x': 80, 'sample_stage_y': 36, 'sample_stage_z': 22.9}]
    
    In [3]: velocities_list                                                                                           
    Out[3]: 
    [{'sample_stage_x': 8.0, 'sample_stage_y': 11.0, 'sample_stage_z': 4.0},
     {'sample_stage_x': 10.9, 'sample_stage_y': 11.0, 'sample_stage_z': 0.9},
     {'sample_stage_x': 11.0, 'sample_stage_y': 4.2, 'sample_stage_z': 0.7}]
    
    In [4]: times_list                                                                                                
    Out[4]: 
    [{'sample_stage_x': 1.8125,
      'sample_stage_y': 1.8181818181818181,
      'sample_stage_z': 1.8049999999999997},
     {'sample_stage_x': 4.541284403669724,
      'sample_stage_y': 4.545454545454546,
      'sample_stage_z': 4.688888888888887},
     {'sample_stage_x': 0.9545454545454546,
      'sample_stage_y': 0.9523809523809523,
      'sample_stage_z': 0.9714285714285711}]
    """


    for param, vel, time_ in zip(params_to_change, velocities_list, times_list):
        hf = HardwareFlyer(params_to_change=param,
                           velocities=vel,
                           time_to_travel=time_,
                           detector=xs, motors=motors)
        yield from bp.fly([hf])

        hf_flyers.append(hf)


def move_back():
    yield from bps.mv(sample_stage.x, 69.5,
                      sample_stage.y, 40,
                      sample_stage.z, 22.22)
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

        # self.detector.settings.acquire.put(0)
        # self.detector.settings.num_images.put(10000)
        # self.detector.settings.acquire_time.put(0.05)
        # self.detector.settings.acquire.put(1)
        start_detector(self.detector)

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
            motor_obj.velocity.put(self.velocities[motor_name])

        for motor_name, motor_obj in self.motors.items():
            if motor_name == slowest_motor:
                self.motor_move_status = motor_obj.set(self.params_to_change[motor_name])
            else:
                motor_obj.set(self.params_to_change[motor_name])

        self.motor_move_status.watch(self._watch_function)

        return NullStatus()

    def complete(self):
        # all motors arrived
        # stop detector
        stop_detector(self.detector)
        return self.motor_move_status

    def describe_collect(self):

        return_dict = {self.name:
                           {
                               # f'{self.name}_{self.detector.channel1.rois.roi01.name}':
                               f'{self.name}_intensity':
                                   # {'source': f'{self.name}_{self.detector.channel1.rois.roi01.name}',
                                   {'source': f'{self.name}_intensity',
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

            # data = {f'{self.name}_{self.detector.channel1.rois.roi01.name}': self.watch_intensities[ind]}
            data = {f'{self.name}_intensity': self.watch_intensities[ind]}
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
        # self.watch_intensities.append(self.detector.channel1.rois.roi01.value.get())
        self.watch_intensities.append(read_detector(self.detector))
        for motor_name, motor_obj in self.motors.items():
            self.watch_positions[motor_name].append(motor_obj.user_readback.get())
        self.watch_timestamps.append(ttime.time())


params_to_change = []



# motors = {sample_stage.x.name: sample_stage.x,
#           sample_stage.y.name: sample_stage.y,
#           sample_stage.z.name: sample_stage.z,}

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
# params_to_change.append({sample_stage.x.name: 72,
#                          sample_stage.y.name: 41,
#                          sample_stage.z.name: 20})
#
# params_to_change.append({sample_stage.x.name: 68,
#                          sample_stage.y.name: 43,
#                          sample_stage.z.name: 19})
#
# params_to_change.append({sample_stage.x.name: 71,
#                          sample_stage.y.name: 39,
#                          sample_stage.z.name: 22.9})

# update function - change params

# RE(bp.fly([hf]))


def calc_velocity(motors, dists, velocity_limits, max_velocity=None, min_velocity=None):
    ret_vels = []
    # check that max_velocity is not None if at least 1 motor doesn't have upper velocity limit
    if any([velocity_limits[i]['high'] == 0 for i in range(len(velocity_limits))]) and max_velocity is None:
        raise ValueError('max_velocity must be set if there is at least 1 motor without upper velocity limit')
    if all([d == 0 for d in dists]):
        # TODO: fix this to handle when motors don't need to move
        # if dists are all 0, set all motors to min velocity
        for i in range(len(velocity_limits)):
            ret_vels.append(velocity_limits[i]['low'])
        return ret_vels
    else:
        # check for negative distances
        if any([d < 0.0 for d in dists]):
            raise ValueError("Distances must be positive. Try using abs()")
        # create list of upper velocity limits for convenience
        upper_velocity_bounds = []
        for j in range(len(velocity_limits)):
            upper_velocity_bounds.append(velocity_limits[j]['high'])
        # find max distances to move and pick the slowest motor of those with max dists
        max_dist_lowest_vel = np.where(dists == np.max(dists))[0]
        max_dist_to_move = -1
        for j in max_dist_lowest_vel:
            if dists[j] >= max_dist_to_move:
                max_dist_to_move = dists[j]
                motor_index_to_use = j
        max_dist_vel = upper_velocity_bounds[motor_index_to_use]
        if max_velocity is not None:
            if max_dist_vel > max_velocity or max_dist_vel == 0:
                max_dist_vel = float(max_velocity)
        time_needed = dists[motor_index_to_use] / max_dist_vel
        for i in range(len(velocity_limits)):
            if i != motor_index_to_use:
                try_vel = np.round(dists[i] / time_needed, 1)
                if try_vel < min_velocity:
                    try_vel = min_velocity
                if try_vel < velocity_limits[i]['low']:
                    try_vel = velocity_limits[i]['low']
                elif try_vel > velocity_limits[i]['high']:
                    if upper_velocity_bounds[i] == 0:
                        pass
                    else:
                        break
                ret_vels.append(try_vel)
            else:
                ret_vels.append(max_dist_vel)
        if len(ret_vels) == len(motors):
            # if all velocities work, return velocities
            return ret_vels
        else:
            # use slowest motor that moves the most
            ret_vels.clear()
            lowest_velocity_motors = np.where(upper_velocity_bounds == np.min(upper_velocity_bounds))[0]
            max_dist_to_move = -1
            for k in lowest_velocity_motors:
                if dists[k] >= max_dist_to_move:
                    max_dist_to_move = dists[k]
                    motor_index_to_use = k
            slow_motor_vel = upper_velocity_bounds[motor_index_to_use]
            if max_velocity is not None:
                if slow_motor_vel > max_velocity or slow_motor_vel == 0:
                    slow_motor_vel = float(max_velocity)
            time_needed = dists[motor_index_to_use] / slow_motor_vel
            for k in range(len(velocity_limits)):
                if k != motor_index_to_use:
                    try_vel = np.round(dists[k] / time_needed, 1)
                    if try_vel < min_velocity:
                        try_vel = min_velocity
                    if try_vel < velocity_limits[k]['low']:
                        try_vel = velocity_limits[k]['low']
                    elif try_vel > velocity_limits[k]['high']:
                        if upper_velocity_bounds[k] == 0:
                            pass
                        else:
                            print("Don't want to be here")
                            raise ValueError("Something terribly wrong happened")
                    ret_vels.append(try_vel)
                else:
                    ret_vels.append(slow_motor_vel)
            return ret_vels


hf_flyers = []


def optimize():
    velocities_list = []
    distances_list = []
    for i, param in enumerate(params_to_change):
        velocities_dict = {}
        distances_dict = {}
        dists = []
        velocity_limits = []
        if i == 0:
            for motor_name, motor_obj in motors.items():
                velocity_limit_dict = {'motor': motor_name,
                                       'low': motor_obj.velocity.low_limit,
                                       'high': motor_obj.velocity.high_limit}
                velocity_limits.append(velocity_limit_dict)
                dists.append(abs(param[motor_name] - motor_obj.user_readback.get()))
        else:
            for motor_name, motor_obj in motors.items():
                velocity_limit_dict = {'motor': motor_name,
                                       'low': motor_obj.velocity.low_limit,
                                       'high': motor_obj.velocity.high_limit}
                velocity_limits.append(velocity_limit_dict)
                dists.append(abs(param[motor_name] - params_to_change[i - 1][motor_name]))
        velocities = calc_velocity(motors.keys(), dists, velocity_limits, max_velocity=10, min_velocity=0)
        if velocities is None:
            velocities = [5.0, 5.0, 5.0]
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
            if vel[motor_name] == 0:
                time_ = 0
            else:
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
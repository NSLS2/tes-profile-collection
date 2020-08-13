import time as ttime

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
    def __init__(self, params_to_change, velocities, time_to_travel,
                 detector, motors):
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

        start_detector(self.detector)

        # Sleep here to avoid issues like this one:
        # Transient Scan ID: 1868     Time: 2020-08-05 13:51:02
        # Persistent Unique Scan ID: '538d3465-a2bf-42b1-b114-11d4580999d2'
        # !!! det reading: 541.0
        # !!! det reading: 541.0
        # !!! det reading: 541.0                                     | 0.0015625/0.5625753 [00:00<00:38, 68.41s/mm]
        # !!! det reading: 541.0▍                                    | 0.0209375/0.5625753 [00:00<00:06, 12.30s/mm]
        # !!! det reading: 40.0███▋                                    | 0.05125/0.5625753 [00:00<00:04,  7.99s/mm]
        # !!! det reading: 40.0█████▋                                 | 0.081875/0.5625753 [00:00<00:03,  6.88s/mm]
        # !!! det reading: 40.0███████▌                              | 0.1121875/0.5625753 [00:00<00:02,  6.37s/mm]
        # !!! det reading: 40.0█████████▌                            | 0.1421875/0.5625753 [00:00<00:02,  6.09s/mm]
        # !!! det reading: 40.0████████████▌                            | 0.1725/0.5625753 [00:01<00:02,  5.89s/mm]
        # !!! det reading: 40.0██████████████▊                          | 0.2025/0.5625753 [00:01<00:02,  5.76s/mm]
        # !!! det reading: 40.0████████████████▉                        | 0.2325/0.5625753 [00:01<00:01,  5.66s/mm]
        # !!! det reading: 40.0█████████████████▊                    | 0.2628125/0.5625753 [00:01<00:01,  5.58s/mm]
        # !!! det reading: 40.0███████████████████▊                  | 0.2928125/0.5625753 [00:01<00:01,  5.53s/mm]
        # !!! det reading: 40.0███████████████████████                | 0.333125/0.5625753 [00:01<00:01,  5.46s/mm]
        # !!! det reading: 40.0█████████████████████████▏             | 0.363125/0.5625753 [00:01<00:01,  5.42s/mm]
        # !!! det reading: 40.0██████████████████████████▌           | 0.3934375/0.5625753 [00:02<00:00,  5.39s/mm]
        # !!! det reading: 40.0██████████████████████████████▏         | 0.42375/0.5625753 [00:02<00:00,  5.36s/mm]
        # !!! det reading: 40.0████████████████████████████████▎       | 0.45375/0.5625753 [00:02<00:00,  5.34s/mm]
        # !!! det reading: 40.0██████████████████████████████████▍     | 0.48375/0.5625753 [00:02<00:00,  5.32s/mm]
        # !!! det reading: 40.0██████████████████████████████████▋   | 0.5140625/0.5625753 [00:02<00:00,  5.30s/mm]
        # !!! det reading: 40.0█████████████████████████████████████▍| 0.5534375/0.5625753 [00:02<00:00,  5.30s/mm]
        # !!! det reading: 40.0████████████████████████████████████████▉| 0.5625/0.5625753 [00:03<00:00,  5.48s/mm]
        # New stream: 'tes_hardware_flyer'
        ttime.sleep(1.0)

        for motor_name, motor_obj in self.motors.items():
            motor_obj.velocity.put(self.velocities[motor_name])

        for motor_name, motor_obj in self.motors.items():
            if motor_name == slowest_motor:
                self.motor_move_status = motor_obj.set(self.params_to_change[motor_name])
            else:
                motor_obj.set(self.params_to_change[motor_name])

        # Call this function once before we start moving all motors to collect the first points.
        self._watch_function()

        self.motor_move_status.watch(self._watch_function)

        return NullStatus()

    def complete(self):
        return self.motor_move_status

    def describe_collect(self):

        return_dict = {self.name:
                       {f'{self.name}_intensity':
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

        return return_dict

    def collect(self):
        # all motors arrived
        stop_detector(self.detector)

        for ind in range(len(self.watch_intensities)):
            motor_dict = {}
            for motor_name, motor_obj in self.motors.items():
                motor_dict.update(
                    {f'{self.name}_{motor_name}_velocity': self.velocities[motor_name],
                     f'{self.name}_{motor_name}_position': self.watch_positions[motor_name][ind]}
                )

            data = {f'{self.name}_intensity': self.watch_intensities[ind]}
            data.update(motor_dict)

            yield {'data': data,
                   'timestamps': {key: self.watch_timestamps[ind] for key in data},
                   'time': self.watch_timestamps[ind],
                   'filled': {key: False for key in data}}

        # # This will produce one event with dictionaries in the <...>_parameters field.
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
        watch_pos, watch_int, watch_time = watch_function(self.motors, self.detector)
        for motor_name in self.motors.keys():
            self.watch_positions[motor_name].extend(watch_pos[motor_name])
        self.watch_intensities.extend(watch_int)
        self.watch_timestamps.extend(watch_time)

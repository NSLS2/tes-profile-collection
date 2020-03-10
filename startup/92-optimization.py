import time as ttime
import bluesky.plans as bp
import bluesky.plan_stubs as bps

from collections import deque

from ophyd.sim import NullStatus, new_uid

import numpy as np
import random


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

        # print('describe_collect:\n', return_dict)

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

            # print('data:\n', data)

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


motors = {sample_stage.x.name: sample_stage.x,
          sample_stage.y.name: sample_stage.y,}
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
# x limits = 45, 55
# y limits = 70, 80
params_to_change.append({sample_stage.x.name: 45,
                         sample_stage.y.name: 79})
                         # sample_stage.z.name: 20})

params_to_change.append({sample_stage.x.name: 53,
                         sample_stage.y.name: 71})
                         # sample_stage.z.name: 19})

params_to_change.append({sample_stage.x.name: 51,
                         sample_stage.y.name: 77})
                         # sample_stage.z.name: 22.9})

params_to_change.append({sample_stage.x.name: 54,
                         sample_stage.y.name: 75})
                         # sample_stage.z.name: 22.9})

params_to_change.append({sample_stage.x.name: 47,
                         sample_stage.y.name: 72})
                         # sample_stage.z.name: 22.9})

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
                try_vel = np.round(dists[i] / time_needed, 5)
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
                    try_vel = np.round(dists[k] / time_needed, 5)
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


def generate_flyer_params(population, max_velocity):
    velocities_list = []
    distances_list = []
    for i, param in enumerate(population):
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
                dists.append(0)
        else:
            for motor_name, motor_obj in motors.items():
                velocity_limit_dict = {'motor': motor_name,
                                       'low': motor_obj.velocity.low_limit,
                                       'high': motor_obj.velocity.high_limit}
                velocity_limits.append(velocity_limit_dict)
                dists.append(abs(param[motor_name] - population[i - 1][motor_name]))
        velocities = calc_velocity(motors.keys(), dists, velocity_limits,
                                   max_velocity=max_velocity, min_velocity=0)
        # velocities = calc_velocity(motors.keys(), dists, velocity_limits, max_velocity=1.3, min_velocity=0)
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
    return velocities_list, times_list


def omea_evaluation(num_of_scans):
    # get the data from databroker
    # num_of_scans is number of db records to look at
    current_fly_data = []
    pop_intensity = []
    pop_positions = []
    max_intensities = []
    max_int_pos = []
    for i in range(-num_of_scans, 0, 1):
        current_fly_data.append(db[i].table('tes_hardware_flyer'))
    for i, t in enumerate(current_fly_data):
        pop_pos_dict = {}
        positions_dict = {}
        max_int_index = t['tes_hardware_flyer_intensity'].idxmax()
        for motor_name in motors.keys():
            positions_dict[motor_name] = t[f'tes_hardware_flyer_{motor_name}_position'][max_int_index]
            pop_pos_dict[motor_name] = t[f'tes_hardware_flyer_{motor_name}_position'][len(t)]
        pop_intensity.append(t['tes_hardware_flyer_intensity'][len(t)])
        max_intensities.append(t['tes_hardware_flyer_intensity'][max_int_index])
        pop_positions.append(pop_pos_dict)
        max_int_pos.append(positions_dict)
    # compare max of each fly scan to population
    # replace population/intensity with higher vals, if they exist
    for i in range(len(max_intensities)):
        if max_intensities[i] > pop_intensity[i]:
            pop_intensity[i] = max_intensities[i]
            for motor_name, pos in max_int_pos[i].items():
                pop_positions[i][motor_name] = pos
    return pop_positions, pop_intensity


best_fitness = [0]
hf_flyers = []
bound_vals = [(45, 55), (70, 80)]
motor_bounds = {}
for i, motor in enumerate(motors.items()):
    motor_bounds[motor[0]] = {'low': bound_vals[i][0], 'high': bound_vals[i][1]}


def optimize(motors=motors, bounds=motor_bounds, max_velocity=0.2, popsize=5, crosspb=.8, mut=.1,
             mut_type='rand/1', threshold=0, max_iter=100):
    # create initial population
    initial_population = []
    for i in range(popsize):
        indv = {}
        if i == 0:
            for motor_name, motor_obj in motors.items():
                indv[motor_name] = motor_obj.user_readback.get()
        else:
            for motor_name, motor_obj in motors.items():
                indv[motor_name] = random.uniform(bounds[motor_name]['low'],
                                                  bounds[motor_name]['high'])
        initial_population.append(indv)
    print('INITIAL POPULATION:', initial_population)

    # velocities_list = []
    # distances_list = []
    # for i, param in enumerate(initial_population):
    #     velocities_dict = {}
    #     distances_dict = {}
    #     dists = []
    #     velocity_limits = []
    #     if i == 0:
    #         for motor_name, motor_obj in motors.items():
    #             velocity_limit_dict = {'motor': motor_name,
    #                                    'low': motor_obj.velocity.low_limit,
    #                                    'high': motor_obj.velocity.high_limit}
    #             velocity_limits.append(velocity_limit_dict)
    #             dists.append(0)
    #     else:
    #         for motor_name, motor_obj in motors.items():
    #             velocity_limit_dict = {'motor': motor_name,
    #                                    'low': motor_obj.velocity.low_limit,
    #                                    'high': motor_obj.velocity.high_limit}
    #             velocity_limits.append(velocity_limit_dict)
    #             dists.append(abs(param[motor_name] - initial_population[i - 1][motor_name]))
    #     velocities = calc_velocity(motors.keys(), dists, velocity_limits, max_velocity=0.2, min_velocity=0)
    #     # velocities = calc_velocity(motors.keys(), dists, velocity_limits, max_velocity=1.3, min_velocity=0)
    #     for motor_name, vel, dist in zip(motors, velocities, dists):
    #         velocities_dict[motor_name] = vel
    #         distances_dict[motor_name] = dist
    #     velocities_list.append(velocities_dict)
    #     distances_list.append(distances_dict)
    #
    # # Validation
    # times_list = []
    # for dist, vel in zip(distances_list, velocities_list):
    #     times_dict = {}
    #     for motor_name, motor_obj in motors.items():
    #         if vel[motor_name] == 0:
    #             time_ = 0
    #         else:
    #             time_ = dist[motor_name] / vel[motor_name]
    #         times_dict[motor_name] = time_
    #     times_list.append(times_dict)
    #
    # print('Distances:  ', distances_list)
    # print('Velocities: ', velocities_list)
    # print('Times: ', times_list)

    velocities_list, times_list = generate_flyer_params(initial_population, max_velocity)

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

    for param, vel, time_ in zip(initial_population, velocities_list, times_list):
        hf = HardwareFlyer(params_to_change=param,
                           velocities=vel,
                           time_to_travel=time_,
                           detector=xs, motors=motors)
        yield from bp.fly([hf])

        hf_flyers.append(hf)

    pop_positions, pop_intensity = omea_evaluation(len(initial_population))

    # # get the data from databroker
    # # need max intensity in between and intensity at each population value
    # current_fly_data = []
    # pop_intensity = []
    # pop_positions = []
    # max_intensities = []
    # max_int_pos = []
    # for i in range(-popsize, 0, 1):
    #     current_fly_data.append(db[i].table('tes_hardware_flyer'))
    # for i, t in enumerate(current_fly_data):
    #     pop_pos_dict = {}
    #     positions_dict = {}
    #     max_int_index = t['tes_hardware_flyer_intensity'].idxmax()
    #     for motor_name in motors.keys():
    #         positions_dict[motor_name] = t[f'tes_hardware_flyer_{motor_name}_position'][max_int_index]
    #         pop_pos_dict[motor_name] = t[f'tes_hardware_flyer_{motor_name}_position'][len(t)]
    #     pop_intensity.append(t['tes_hardware_flyer_intensity'][len(t)])
    #     max_intensities.append(t['tes_hardware_flyer_intensity'][max_int_index])
    #     pop_positions.append(pop_pos_dict)
    #     max_int_pos.append(positions_dict)
    # print('pop_positions', pop_positions)
    # print('max_int_pos', max_int_pos)
    # # compare max of each fly scan to population
    # # replace population/intensity with higher vals, if they exist
    # for i in range(len(max_intensities)):
    #     if max_intensities[i] > pop_intensity[i]:
    #         pop_intensity[i] = max_intensities[i]
    #         for motor_name, pos in max_int_pos[i].items():
    #             pop_positions[i][motor_name] = pos
    # print('New positions:', pop_positions)
    # print('New intensities:', pop_intensity)

    # Termination conditions
    v = 0  # generation number
    consec_best_ctr = 0  # counting successive generations with no change to best value
    old_best_fit_val = 0
    # while not v > 0:
    while not ((v > max_iter) or (consec_best_ctr >= 5 and old_best_fit_val >= threshold)):
        print(f'GENERATION {v + 1}')
        best_gen_sol = []
        # mutate
        mutated_trial_pop = mutate(pop_positions, mut_type, mut, bounds, ind_sol=pop_intensity)
        # crossover
        cross_trial_pop = crossover(pop_positions, mutated_trial_pop, crosspb)

        # select, how can this be it's own function?
        select_positions = [elm for elm in cross_trial_pop]
        indv = {}
        for motor_name, motor_obj in motors.items():
            indv[motor_name] = motor_obj.user_readback.get()
        select_positions.insert(0, indv)
        velocities_list, times_list = generate_flyer_params(select_positions, max_velocity)
        for param, vel, time_ in zip(select_positions, velocities_list, times_list):
            hf = HardwareFlyer(params_to_change=param,
                               velocities=vel,
                               time_to_travel=time_,
                               detector=xs, motors=motors)
            yield from bp.fly([hf])

            hf_flyers.append(hf)
        positions, intensities = omea_evaluation(len(select_positions))
        positions = positions[1:]
        intensities = intensities[1:]
        for i in range(len(intensities)):
            if intensities[i] > pop_intensity[i]:
                pop_positions[i] = positions[i]
                pop_intensity[i] = intensities[i]
        # pop_positions, pop_intensity = select(positions, intensities, motors,
        #                                       cross_trial_pop, max_velocity)

        # get best solution
        gen_best = np.max(pop_intensity)
        best_indv = pop_positions[pop_intensity.index(gen_best)]
        best_gen_sol.append(best_indv)
        best_fitness.append(gen_best)

        print('      > FITNESS:', gen_best)
        print('         > BEST POSITIONS:', best_indv)

        v += 1
        if np.round(gen_best, 6) == np.round(old_best_fit_val, 6):
            consec_best_ctr += 1
            print('Counter:', consec_best_ctr)
        else:
            consec_best_ctr = 0
        old_best_fit_val = gen_best

        if consec_best_ctr >= 5 and old_best_fit_val >= threshold:
            print('Finished')
            break
        else:
            # randomize worst individual and repeat from mutate
            pos_to_check = []
            change_indx = pop_intensity.index(np.min(pop_intensity))
            changed_indv = pop_positions[change_indx]
            indv = {}
            for motor_name, motor_obj in motors.items():
                indv[motor_name] = motor_obj.user_readback.get()
            pos_to_check.append(indv)
            indv = {}
            for motor_name, pos in changed_indv.items():
                indv[motor_name] = random.uniform(bounds[motor_name]['low'],
                                                  bounds[motor_name]['high'])
            pos_to_check.append(indv)

            velocities_list, times_list = generate_flyer_params(pos_to_check, max_velocity)
            for param, vel, time_ in zip(pos_to_check, velocities_list, times_list):
                hf = HardwareFlyer(params_to_change=param,
                                   velocities=vel,
                                   time_to_travel=time_,
                                   detector=xs, motors=motors)
                yield from bp.fly([hf])

                hf_flyers.append(hf)
            rand_position, rand_intensity = omea_evaluation(len(pos_to_check))
            rand_position = rand_position[1:]
            rand_intensity = rand_intensity[1:]
            if rand_intensity[0] > pop_intensity[change_indx]:
                pop_positions[change_indx] = rand_position[0]
                pop_intensity[change_indx] = rand_intensity[0]

    # best solution overall should be last one
    x_best = best_gen_sol[-1]
    print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
    print('It took', v, 'generations')

    plot_index = np.arange(len(best_fitness))
    plt.figure()
    plt.plot(plot_index, best_fitness)


def ensure_bounds(vec, bounds):
    # Makes sure each individual stays within bounds and adjusts them if they aren't
    vec_new = {}
    # cycle through each variable in vector
    for motor_name, pos in vec.items():
        # variable exceeds the minimum boundary
        if pos < bounds[motor_name]['low']:
            vec_new[motor_name] = bounds[motor_name]['low']
        # variable exceeds the maximum boundary
        if pos > bounds[motor_name]['high']:
            vec_new[motor_name] = bounds[motor_name]['high']
        # the variable is fine
        if bounds[motor_name]['low'] <= pos <= bounds[motor_name]['high']:
            vec_new[motor_name] = pos
    return vec_new


def rand_1(pop, popsize, target_indx, mut, bounds):
    # mutation strategy
    # v = x_r1 + F * (x_r2 - x_r3)
    idxs = [idx for idx in range(popsize) if idx != target_indx]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]

    x_diff = {}
    for motor_name, pos in x_2.items():
        x_diff[motor_name] = x_2[motor_name] - x_3[motor_name]
    v_donor = {}
    for motor_name, pos in x_1.items():
        v_donor[motor_name] = x_1[motor_name] + mut * x_diff[motor_name]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def mutate(population, strategy, mut, bounds, ind_sol):
    mutated_indv = []
    for i in range(len(population)):
        if strategy == 'rand/1':
            v_donor = rand_1(population, len(population), i, mut, bounds)
        # elif strategy == 'best/1':
        #     v_donor = best_1(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'current-to-best/1':
        #     v_donor = current_to_best_1(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'best/2':
        #     v_donor = best_2(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'rand/2':
        #     v_donor = rand_2(population, len(population), i, mut, bounds)
        mutated_indv.append(v_donor)
    return mutated_indv


def crossover(population, mutated_indv, crosspb):
    crossover_indv = []
    for i in range(len(population)):
        x_t = population[i]
        v_trial = {}
        for motor_name, pos in x_t.items():
            crossover_val = random.random()
            if crossover_val <= crosspb:
                v_trial[motor_name] = mutated_indv[i][motor_name]
            else:
                v_trial[motor_name] = x_t[motor_name]
        crossover_indv.append(v_trial)
    return crossover_indv


def select(population, ind_sol, motors, crossover_indv, max_velocity):
    positions = [elm for elm in crossover_indv]
    positions.insert(0, population[0])
    velocities_list, times_list = generate_flyer_params(population, max_velocity)
    for param, vel, time_ in zip(population, velocities_list, times_list):
        hf = HardwareFlyer(params_to_change=param,
                           velocities=vel,
                           time_to_travel=time_,
                           detector=xs, motors=motors)
        yield from bp.fly([hf])

        hf_flyers.append(hf)

    positions, intensities = omea_evaluation(len(population))
    positions = positions[1:]
    intensities = intensities[1:]
    for i in range(len(intensities)):
        if intensities[i] > ind_sol[i]:
            population[i] = positions[i]
            ind_sol[i] = intensities[i]
    return population, ind_sol


def move_back():
    yield from bps.mv(sample_stage.x, 50,
                      sample_stage.y, 75)

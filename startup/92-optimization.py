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
        rd = read_detector(self.detector)
        self.watch_intensities.append(rd)
        for motor_name, motor_obj in self.motors.items():
            self.watch_positions[motor_name].append(motor_obj.user_readback.get())
        self.watch_timestamps.append(ttime.time())


motor_dict = {sample_stage.x.name: sample_stage.x,
              sample_stage.y.name: sample_stage.y,
              # sample_stage.z.name: sample_stage.z,
              }

bound_vals = [(43, 44), (34, 35)]
motor_bounds = {}
for i, motor in enumerate(motor_dict.items()):
    motor_bounds[motor[0]] = {'low': bound_vals[i][0], 'high': bound_vals[i][1]}

# Note: run it with
# RE(optimize(run_hardware_fly, motors=motor_dict, detector=xs, bounds=motor_bounds))

# TODO: change motor list to be dict of dicts;
#  {motor_name: {position: val}}, {motor_name: {bounds: [low, high]}}

# TODO: merge "params_to_change" and "velocities" lists of dictionaries to become lists of dicts of dicts.


def calc_velocity(motors, dists, velocity_limits, max_velocity=None, min_velocity=None):
    """
    Calculates velocities of all motors

    Velocities calculated will allow motors to approximately start and stop together

    Parameters
    ----------
    motors : dict
             In the format {motor_name: motor_object}
             Ex. {sample_stage.x.name: sample_stage.x}
    dists :
    velocity_limits :
    max_velocity : float
                   Set this to limit the absolute highest velocity of any motor
    min_velocity : float
                   Set this to limit the absolute lowest velocity of any motor

    Returns
    -------
    ret_vels : list
               List of velocities for each motor
    """
    ret_vels = []
    # check that max_velocity is not None if at least 1 motor doesn't have upper velocity limit
    if any([lim['high'] == 0 for lim in velocity_limits]) and max_velocity is None:
        vel_max_lim_0 = []
        for lim in velocity_limits:
            if lim['high'] == 0:
                vel_max_lim_0.append(lim['motor'])
        raise ValueError(f'The following motors have unset max velocity limits: {vel_max_lim_0}. '
                         f'max_velocity must be set')
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
            lowest_velocity_motors = np.where(upper_velocity_bounds ==
                                              np.min(upper_velocity_bounds))[0]
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


def run_hardware_fly(motors, detector, population, max_velocity, min_velocity):
    uid_list = []
    flyers = generate_flyers(motors=motors, detector=detector, population=population,
                             max_velocity=max_velocity, min_velocity=min_velocity)
    for flyer in flyers:
        yield from bp.fly([flyer])
    for i in range(-len(flyers), 0):
        uid_list.append(i)
    # uid = (yield from bp.fly([hf]))
    # uid_list.append(uid)
    return uid_list


def generate_flyers(motors, detector, population, max_velocity, min_velocity):
    hf_flyers = []
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
        velocities = calc_velocity(motors=motors.keys(), dists=dists, velocity_limits=velocity_limits,
                                   max_velocity=max_velocity, min_velocity=min_velocity)
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

    for param, vel, time_ in zip(population, velocities_list, times_list):
        hf = HardwareFlyer(params_to_change=param,
                           velocities=vel,
                           time_to_travel=time_,
                           detector=detector,
                           motors=motors)
        hf_flyers.append(hf)
    return hf_flyers


def omea_evaluation(motors, uids, flyer_name, intensity_name, field_name):
    # get the data from databroker
    current_fly_data = []
    pop_intensity = []
    pop_positions = []
    max_intensities = []
    max_int_pos = []
    for uid in uids:
        current_fly_data.append(db[uid].table(flyer_name))
    for i, t in enumerate(current_fly_data):
        pop_pos_dict = {}
        positions_dict = {}
        max_int_index = t[f'{flyer_name}_{intensity_name}'].idxmax()
        for param_name in motors.keys():
            positions_dict[param_name] = t[f'{flyer_name}_{param_name}_{field_name}'][max_int_index]
            pop_pos_dict[param_name] = t[f'{flyer_name}_{param_name}_{field_name}'][len(t)]
        pop_intensity.append(t[f'{flyer_name}_{intensity_name}'][len(t)])
        max_intensities.append(t[f'{flyer_name}_{intensity_name}'][max_int_index])
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


def optimize(fly_plan, motors, detector, bounds, max_velocity=0.2, min_velocity=0,
             popsize=5, crosspb=.8, mut=.1, mut_type='rand/1', threshold=0, max_iter=100,
             flyer_name='tes_hardware_flyer', intensity_name='intensity', field_name='position'):
    """
    Optimize beamline using hardware flyers and differential evolution

    Custom plan to optimize motor positions of the TES beamline using differential evolution

    Parameters
    ----------
    fly_plan : callable
               Fly scan plan for current type of flyer.
               Currently the only option is `run_hardware_fly`, but another will be added for sirepo simulations
    motors : dict
             Keys are motor names and values are motor objects
    detector : detector object or None
               Detector to use, or None if no detector will be used
    bounds : dict of dicts
             Keys are motor names and values are dicts of low and high bounds. See format below.
             {'motor_name': {'low': lower_bound, 'high': upper_bound}}
    max_velocity : float, optional
                   Absolute maximum velocity for all motors
                   Default is 0.2
    min_velocity : float, optional
                   Absolute minumum velocity for all motors
    popsize : int, optional
              Size of population
    crosspb : float, optional
              Probability of crossover. Must be in range [0, 1]
    mut : float, optional
          Mutation factor. Must be in range [0, 1]
    mut_type : {'rand/1', 'best/1'}, optional
               Mutation strategy to use. 'rand/1' chooses random individuals to compare to.
               'best/1' uses the best individual to compare to.
               Default is 'rand/1'
    threshold : float, optional
                Threshold that intensity must be greater than or equal to to stop execution
    max_iter : int, optional
               Maximum iterations to allow
    flyer_name : str, optional
                 Name of flyer. DataBroker stream name
                 Default is 'tes_hardware_flyer'
    intensity_name : {'intensity', 'mean'}, optional
                     Hardware optimization would use 'intensity'. Sirepo optimization would use 'mean'
                     Default is 'intensity'
    field_name : str, optional
                 Default is 'position'
    """
    # This disables live plots, not needed for this plan.
    bec.disable_plots()

    global optimized_positions
    # check if bounds passed in are within the actual bounds of the motors
    check_opt_bounds(motors, bounds)
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
    uid_list = (yield from fly_plan(motors=motors, detector=detector, population=initial_population,
                                    max_velocity=max_velocity, min_velocity=min_velocity))
    pop_positions, pop_intensity = omea_evaluation(motors=motors, uids=uid_list,
                                                   flyer_name=flyer_name, intensity_name=intensity_name,
                                                   field_name=field_name)
    # Termination conditions
    v = 0  # generation number
    consec_best_ctr = 0  # counting successive generations with no change to best value
    old_best_fit_val = 0
    best_fitness = [0]
    while not ((v > max_iter) or (consec_best_ctr >= 5 and old_best_fit_val >= threshold)):
        print(f'GENERATION {v + 1}')
        best_gen_sol = []
        # mutate
        mutated_trial_pop = mutate(population=pop_positions, strategy=mut_type, mut=mut,
                                   bounds=bounds, ind_sol=pop_intensity)
        # crossover
        cross_trial_pop = crossover(population=pop_positions, mutated_indv=mutated_trial_pop,
                                    crosspb=crosspb)
        # select
        select_positions = create_selection_params(motors=motors, cross_indv=cross_trial_pop)
        uid_list = (yield from fly_plan(motors=motors, detector=detector, population=select_positions,
                                        max_velocity=max_velocity, min_velocity=min_velocity))

        pop_positions, pop_intensity = select(population=pop_positions, intensities=pop_intensity,
                                              motors=motors, uids=uid_list, flyer_name=flyer_name,
                                              intensity_name=intensity_name, field_name=field_name)

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
            positions, change_indx = create_rand_selection_params(motors=motors, intensities=pop_intensity,
                                                                  bounds=bounds)
            uid_list = (yield from fly_plan(motors=motors, detector=detector, population=positions,
                                            max_velocity=max_velocity, min_velocity=min_velocity))
            rand_pop, rand_int = select(population=[pop_positions[change_indx]],
                                        intensities=[pop_intensity[change_indx]],
                                        motors=motors, uids=uid_list, flyer_name=flyer_name,
                                        intensity_name=intensity_name, field_name=field_name)
            assert len(rand_pop) == 1 and len(rand_int) == 1
            pop_positions[change_indx] = rand_pop[0]
            pop_intensity[change_indx] = rand_int[0]

    # best solution overall should be last one
    x_best = best_gen_sol[-1]
    optimized_positions = x_best
    print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
    print('It took', v, 'generations')

    print('Moving to optimal positions')
    yield from move_to_optimized_positions(motors, optimized_positions)

    # Enable live plots here to be available for other plans.
    bec.enable_plots()


    plot_index = np.arange(len(best_fitness))
    plt.figure()
    plt.plot(plot_index, best_fitness)


def check_opt_bounds(motors, bounds):
    for motor_name, bound in bounds.items():
        if bound['low'] > bound['high']:
            raise ValueError(f"Invalid bounds for {motor_name}. Current bounds are set to "
                             f"{bound['low'],bound['high']}, but lower bound is greater "
                             f"than upper bound")
        if bound['low'] < motors[motor_name].low_limit or bound['high']\
                > motors[motor_name].high_limit:
            raise ValueError(f"Invalid bounds for {motor_name}. Current bounds are set to "
                             f"{bound['low'],bound['high']}, but {motor_name} has bounds of "
                             f"{motors[motor_name].limits}")


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
    v_donor = ensure_bounds(vec=v_donor, bounds=bounds)
    return v_donor


def best_1(pop, popsize, target_indx, mut, bounds, ind_sol):
    # mutation strategy
    # v = x_best + F * (x_r1 - x_r2)
    x_best = pop[ind_sol.index(np.max(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != target_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]

    x_diff = {}
    for motor_name, pos in x_1.items():
        x_diff[motor_name] = x_1[motor_name] - x_2[motor_name]
    v_donor = {}
    for motor_name, pos in x_best.items():
        v_donor[motor_name] = x_best[motor_name] + mut * x_diff[motor_name]
    v_donor = ensure_bounds(vec=v_donor, bounds=bounds)
    return v_donor


def mutate(population, strategy, mut, bounds, ind_sol):
    mutated_indv = []
    for i in range(len(population)):
        if strategy == 'rand/1':
            v_donor = rand_1(pop=population, popsize=len(population), target_indx=i,
                             mut=mut, bounds=bounds)
        elif strategy == 'best/1':
            v_donor = best_1(pop=population, popsize=len(population), target_indx=i,
                             mut=mut, bounds=bounds, ind_sol=ind_sol)
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


def create_selection_params(motors, cross_indv):
    positions = [elm for elm in cross_indv]
    indv = {}
    for motor_name, motor_obj in motors.items():
        indv[motor_name] = motor_obj.user_readback.get()
    positions.insert(0, indv)
    return positions


def create_rand_selection_params(motors, intensities, bounds):
    positions = []
    change_indx = intensities.index(np.min(intensities))
    indv = {}
    for motor_name, motor_obj in motors.items():
        indv[motor_name] = motor_obj.user_readback.get()
    positions.append(indv)
    indv = {}
    for motor_name, bound in bounds.items():
        indv[motor_name] = random.uniform(bound['low'], bound['high'])
    positions.append(indv)
    return positions, change_indx


def select(population, intensities, motors, uids, flyer_name, intensity_name, field_name):
    new_population, new_intensities = omea_evaluation(motors=motors, uids=uids, flyer_name=flyer_name,
                                                      intensity_name=intensity_name, field_name=field_name)
    del new_population[0]
    del new_intensities[0]
    assert len(new_population) == len(population)
    for i in range(len(new_intensities)):
        if new_intensities[i] > intensities[i]:
            population[i] = new_population[i]
            intensities[i] = new_intensities[i]
    return population, intensities


def move_to_optimized_positions(motors, opt_pos):
    """Move motors to best positions"""
    mv_params = []
    for motor_obj, pos in zip(motors.values(), opt_pos.values()):
        # yield from bps.mv(motor_obj, pos)
        mv_params.append(motor_obj)
        mv_params.append(pos)
    yield from bps.mv(*mv_params)


# Logbook: 2020-08-05

# Transient Scan ID: 2106     Time: 2020-08-05 14:33:11
# Persistent Unique Scan ID: '7744d3c6-7640-42e7-ae43-a404069cc181'
# !!! det reading: 110.0
# !!! det reading: 110.0
# !!! det reading: 110.0                                                                                                      | 0.0025/0.3955 [00:00<00:11, 28.91s/mm]
# !!! det reading: 66.0███████▏                                                                                            | 0.0284375/0.3955 [00:00<00:02,  7.97s/mm]
# !!! det reading: 71.0██████████████▏                                                                                     | 0.0559375/0.3955 [00:00<00:02,  6.81s/mm]
# !!! det reading: 37.0█████████████████████                                                                               | 0.0834375/0.3955 [00:00<00:02,  6.42s/mm]
# !!! det reading: 149.0███████████████████████████                                                                        | 0.1109375/0.3955 [00:00<00:01,  6.21s/mm]
# !!! det reading: 343.0██████████████████████████████████                                                                 | 0.1384375/0.3955 [00:00<00:01,  6.09s/mm]
# !!! det reading: 230.0█████████████████████████████████████████▉                                                           | 0.16625/0.3955 [00:00<00:01,  6.00s/mm]
# !!! det reading: 196.0████████████████████████████████████████████████▉                                                    | 0.19375/0.3955 [00:01<00:01,  5.94s/mm]
# !!! det reading: 89.0█████████████████████████████████████████████████████████                                             | 0.22125/0.3955 [00:01<00:01,  5.90s/mm]
# !!! det reading: 79.0██████████████████████████████████████████████████████████████████▋                                   | 0.25875/0.3955 [00:01<00:00,  5.85s/mm]
# !!! det reading: 201.0████████████████████████████████████████████████████████████████████████▊                            | 0.28625/0.3955 [00:01<00:00,  5.83s/mm]
# !!! det reading: 60.0████████████████████████████████████████████████████████████████████████████████▉                     | 0.31375/0.3955 [00:01<00:00,  5.81s/mm]
# !!! det reading: 124.0███████████████████████████████████████████████████████████████████████████████████████              | 0.34125/0.3955 [00:01<00:00,  5.79s/mm]
# !!! det reading: 156.0██████████████████████████████████████████████████████████████████████████████████████████████       | 0.36875/0.3955 [00:02<00:00,  5.77s/mm]
# !!! det reading: 205.0████████████████████████████████████████████████████████████████████████████████████████████████████▌| 0.39375/0.3955 [00:02<00:00,  5.80s/mm]
# !!! det reading: 205.025mm [00:02,  6.33s/mm]
# New stream: 'tes_hardware_flyer'

# In [7]: hdr = db['7744d3c6-7640-42e7-ae43-a404069cc181']
#
# In [8]: hdr.table(stream_name='tes_hardware_flyer', fields=['tes_hardware_flyer_sample_stage_x_position', 'tes_hardware_flyer_sample_stage_y_position', 'tes_hardwar
#    ...: e_flyer_intensity'])
# Out[8]:
#                                  time  tes_hardware_flyer_intensity  tes_hardware_flyer_sample_stage_x_position  tes_hardware_flyer_sample_stage_y_position
# seq_num
# 1       2020-08-05 14:33:14.213995934                         205.0                                   54.365938                                   40.369062
#
# In [9]:

# ^^^ solved by rewriting the watch function (no globals)

# TODO:
# 1) have a live plot for the fitness (convergence plot)
# 2) improve the algorithm to optimize the travel path (should not go back to the same point)
# 3) Usability:
#    - convenient I/O to the saved data (pandas dataframes, sorting, ...)
#    - clean up prints and better output log
#    - add metadata about the samples
#    - plot trajectory


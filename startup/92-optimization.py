import numpy as np
import random


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

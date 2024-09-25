print(f"Loading {__file__!r} ...")

import matplotlib.pyplot as plt
import numpy as np
from random import random, uniform
import time
import copy

from ophyd import EpicsMotor, Device, Component as Cpt

'''
    Give PV name without having to create the class?
'''


class TestingStage(Device):
    x = Cpt(EpicsMotor, "1}Mtr")


class SampleStage(Device):
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")
    z = Cpt(EpicsMotor, "Z}Mtr")


testing_stage = TestingStage(prefix='XF:08BM{MC:07-Ax:', name='testing_stage')
sample_stage = SampleStage(prefix='XF:08BMES-OP{SM:1-Ax:', name='sample_stage')

global positions
global intensities

motor_list = [sample_stage.x, sample_stage.y]  # , sample_stage.z]
limits = [(50, 60), (50, 60)]
best_gen_sol = []  # holding best individuals of each generation
best_fitness = [0]  # holds fitness of best individuals of each generation


def test_velocity_using_time(m_pos, m_set, t):
    motors = [sample_stage.x, sample_stage.y, sample_stage.z]
    for i in range(len(motors)):
        motors[i].move(m_pos[i])
    moving = []
    for i in range(len(motors)):
        moving.append(np.abs(m_set[i] - motors[i].position))
    # set velocity to distance / time (t)
    for i in range(len(motors)):
        velocity = np.round(moving[i] / t, 1)
        if motors[i].velocity.low_limit <= velocity <= motors[i].velocity.high_limit:
            motors[i].velocity.set(velocity)
            print(velocity)
        else:
            print('Bad velocity')
    for i in range(len(motors)):
        motors[i].set(m_set[i])
    print('done')


def test_motor_reading(pos):
    # test function
    position_list = []
    status = testing_stage.x.move(pos, wait=False)
    while not status.done:
        position_list.append(testing_stage.x.position)
    num_of_positions = len(position_list)
    return position_list, num_of_positions


def plot_test_motor_reading(position_list):
    # test function
    pos_index = np.arange(len(position_list))
    plt.figure()
    plt.plot(pos_index, position_list)


def simple_parabola(x):
    # function to optimize
    x = np.asarray(x)
    return -3 * x ** 2 + 3


def beamline_test_function(x):
    # function to optimize
    x = np.asarray(x)
    return np.sin(4 * x) - np.cos(8 * x) + 2


def ensure_bounds(vec, bounds):
    # Makes sure each individual stays within bounds and adjusts them if they aren't
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):
        # variable exceeds the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])
        # variable exceeds the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])
        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
    return vec_new


def omea(population, motors):
    ind_sol = []
    positions = []
    intensities = []
    watch_positions = []
    watch_intensities = []
    movements = []

    def f(*args, **kwargs):
        curr_pos = []
        for jj in range(len(motors)):
            read_val = motors[jj].user_readback.get()
            curr_pos.append(read_val)
        watch_positions.append(curr_pos)
        watch_intensities.append(xs.channel1.rois.roi01.value.get())

    print('Evaluating individuals\nProgress:')
    print(str(1) + ' of ' + str(len(population)))
    movements.clear()
    # move all motors to the first individual
    for i in range(len(motors)):
        movements.append(np.abs(motors[i].position - population[0][i]))
    max_move_index = movements.index(np.max(movements))

    for i in range(len(motors)):
        # set motors to move to next individual
        if i == max_move_index:
            st = motors[i].set(population[0][i])
        else:
            motors[i].set(population[0][i])
    st.watch(f)  # use status on motor that needs to move the most
    while not st.done:
        time.sleep(0.00001)
    # get intensity
    ind_sol.append(xs.channel1.rois.roi01.value.get())

    for i in range(1, len(population)):
        # now go through each individual and do OMEA
        old_population = copy.deepcopy(population)  # keeps track of where to move next
        unique_between = []
        unique_eval = []
        positions.clear()
        intensities.clear()
        watch_positions.clear()
        watch_intensities.clear()
        print(str(i + 1) + ' of ' + str(len(population)))

        curr_pos = []
        movements.clear()
        for jj in range(len(motors)):
            read_val = motors[jj].user_readback.get()
            curr_pos.append(read_val)
            movements.append(np.abs(motors[jj].position - old_population[i][jj]))
        max_move_index = movements.index(np.max(movements))

        # change velocities before movement
        update_velocity(motors, movements)

        watch_positions.append(curr_pos)
        watch_intensities.append(xs.channel1.rois.roi01.value.get())

        for j in range(len(motors)):
            # set motors to move to next individual
            if j == max_move_index:
                st = motors[j].set(old_population[i][j])
            else:
                motors[j].set(old_population[i][j])
        st.watch(f)  # use status on motor that needs to move the most
        while not st.done:
            time.sleep(0.00001)
        # fitness of next individual
        ind_sol.append(xs.channel1.rois.roi01.value.get())

        positions = np.array(watch_positions)
        positions = positions.reshape((positions.shape[0], len(motors))).tolist()
        intensities = np.array(watch_intensities).tolist()

        for j in range(len(positions)):  # ***
            # gets unique positions
            if positions[j] not in unique_between:
                unique_between.append(positions[j])
                unique_eval.append(intensities[j])
        # cut out first and last elements (already accounted for)
        between = unique_between[1:-1]
        between_eval = unique_eval[1:-1]

        # find index of max if values were found in between individuals
        try:
            ii = between_eval.index(np.max(between_eval))
            # update population and individual solutions (ind_sol)
            if between_eval[ii] > ind_sol[i]:
                ind_sol[i] = between_eval[ii]
                for k in range(len(population[i])):
                    population[i][k] = between[ii][k]

        except ValueError:
            # this means nothing was found between individuals
            # individuals are very close together or the same value
            pass

    return population, ind_sol


def rand_1(pop, popsize, t_indx, mut, bounds):
    # mutation strategy
    # v = x_r1 + F * (x_r2 - x_r3)
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]

    x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    v_donor = [x_1_i + mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def best_1(pop, popsize, t_indx, mut, bounds, ind_sol):
    # mutation strategy
    # v = x_best + F * (x_r1 - x_r2)
    x_best = pop[ind_sol.index(np.max(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]

    x_diff = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    v_donor = [x_b + mut * x_diff_i for x_b, x_diff_i in zip(x_best, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def current_to_best_1(pop, popsize, t_indx, mut, bounds, ind_sol):
    # mutation strategy
    # v = x_curr + F * (x_best - x_curr) + F * (x_r1 - r_r2)
    x_best = pop[ind_sol.index(np.max(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_curr = pop[t_indx]

    x_diff1 = [x_b - x_c for x_b, x_c in zip(x_best, x_curr)]
    x_diff2 = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    v_donor = [x_c + mut * x_diff_1 + mut * x_diff_2 for x_c, x_diff_1, x_diff_2
               in zip(x_curr, x_diff1, x_diff2)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def best_2(pop, popsize, t_indx, mut, bounds, ind_sol):
    # mutation strategy
    # v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - r_r4)
    x_best = pop[ind_sol.index(np.max(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b, c, d = np.random.choice(idxs, 4, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]
    x_4 = pop[d]

    x_diff1 = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    x_diff2 = [x_3_i - x_4_i for x_3_i, x_4_i in zip(x_3, x_4)]
    v_donor = [x_b + mut * x_diff_1 + mut * x_diff_2 for x_b, x_diff_1, x_diff_2
               in zip(x_best, x_diff1, x_diff2)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def rand_2(pop, popsize, t_indx, mut, bounds):
    # mutation strategy
    # v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - r_r5)
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b, c, d, e = np.random.choice(idxs, 5, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]
    x_4 = pop[d]
    x_5 = pop[e]

    x_diff1 = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    x_diff2 = [x_4_i - x_5_i for x_4_i, x_5_i in zip(x_4, x_5)]
    v_donor = [x_1_i + mut * x_diff_1 + mut * x_diff_2 for x_1_i, x_diff_1, x_diff_2
               in zip(x_1, x_diff1, x_diff2)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def test_velocity_using_motor_moving_most(m_pos, m_set):
    motors = [sample_stage.x, sample_stage.y, sample_stage.z]
    for i in range(len(motors)):
        motors[i].move(m_pos[i])
    moving = []
    for i in range(len(motors)):
        moving.append(np.abs(m_set[i] - motors[i].position))
    # set velocity to distance / time (t)
    max_to_move = np.max(moving)
    max_moving_motor_index = moving.index(max_to_move)
    motors[max_moving_motor_index].velocity.set(motors[max_moving_motor_index].velocity.high_limit)
    time_needed = max_to_move / motors[max_moving_motor_index].velocity.high_limit
    for i in range(len(motors)):
        if i != max_moving_motor_index:
            velocity = np.round(moving[i] / time_needed, 1)
            if motors[i].velocity.low_limit <= velocity <= motors[i].velocity.high_limit:
                motors[i].velocity.set(velocity)
            else:
                print("This is a problem that needs thinking and fixing")
    for i in range(len(motors)):
        motors[i].set(m_set[i])
    return


def update_velocity(motors, distances_to_move):
    # call before any movement
    max_distance = np.max(distances_to_move)
    max_dist_index = distances_to_move.index(max_distance)
    motors[max_dist_index].velocity.set(motors[max_dist_index].velocity.high_limit)  # ***
    time_needed = max_distance / motors[max_dist_index].velocity.get()
    for i in range(len(motors)):
        if i != max_dist_index:
            velocity = np.round(distances_to_move[i] / time_needed, 1)
            if motors[i].velocity.low_limit <= velocity <= motors[i].velocity.high_limit:
                motors[i].velocity.set(velocity)
            else:
                if velocity < motors[i].velocity.low_limit:
                    motors[i].velocity.set(motors[i].velocity.low_limit)
                elif velocity > motors[i].velocity.high_limit:
                    motors[i].velocity.set(motors[i].velocity.high_limit)


def mutate(population, strategy, mut, bounds, ind_sol):
    mutated_indv = []
    for i in range(len(population)):
        if strategy == 'rand/1':
            v_donor = rand_1(population, len(population), i, mut, bounds)
        elif strategy == 'best/1':
            v_donor = best_1(population, len(population), i, mut, bounds, ind_sol)
        elif strategy == 'current-to-best/1':
            v_donor = current_to_best_1(population, len(population), i, mut, bounds, ind_sol)
        elif strategy == 'best/2':
            v_donor = best_2(population, len(population), i, mut, bounds, ind_sol)
        elif strategy == 'rand/2':
            v_donor = rand_2(population, len(population), i, mut, bounds)
        mutated_indv.append(v_donor)
    return mutated_indv


def crossover(population, mutated_indv, crosspb):
    crossover_indv = []
    for i in range(len(population)):
        v_trial = []
        x_t = population[i]
        for j in range(len(x_t)):
            crossover_val = random()
            if crossover_val <= crosspb:
                v_trial.append(mutated_indv[i][j])
            else:
                v_trial.append(x_t[j])
        crossover_indv.append(v_trial)
    return crossover_indv


def select(population, crossover_indv, ind_sol, motors):
    positions = [elm for elm in crossover_indv]
    positions.insert(0, population[0])
    positions, evals = omea(positions, motors)
    positions = positions[1:]
    evals = evals[1:]
    for i in range(len(evals)):
        if evals[i] < ind_sol[i]:
            population[i] = positions[i]
            ind_sol[i] = evals[i]
    population.reverse()
    ind_sol.reverse()
    return population, ind_sol


def diff_ev(motors, threshold, bounds=None, popsize=10, crosspb=0.8, mut=0.05, mut_type='rand/1'):
    if bounds is None:
        bounds = []
        for i in range(len(motors)):
            bounds.append((motor_list[i].low_limit, motor_list[i].high_limit))
    xs.settings.acquire.put(0)
    xs.settings.num_images.put(10000)
    xs.settings.acquire_time.put(0.05)
    xs.settings.acquire.put(1)

    # Initial population
    population = []
    init_indv = []
    movements = []
    # gets initial position of motors
    for i in range(len(motors)):
        init_indv.append(motors[i].position)
    population.append(init_indv)
    # randomize the rest of the population
    for i in range(popsize - 1):
        indv = []
        for j in range(len(bounds)):
            indv.append(uniform(bounds[j][0], bounds[j][1]))
        population.append(indv)
    init_pop = population[:]

    # evaluate fitness of individuals
    pop, ind_sol = omea(init_pop, motors)
    pop.reverse()
    ind_sol.reverse()

    # Termination conditions
    v = 0  # generation number
    consec_best_ctr = 0  # counting successive generations with no change to best value
    old_best_fit_val = 0
    while not (consec_best_ctr >= 5 and old_best_fit_val >= threshold):
        print('\nGENERATION ' + str(v + 1))
        best_gen_sol = []  # score keeping
        print('Performing mutation, crossover, and selection')  # ***
        mutated_trial_pop = mutate(pop, mut_type, mut, bounds, ind_sol)
        cross_trial_pop = crossover(pop, mutated_trial_pop, crosspb)
        pop, ind_sol = select(pop, cross_trial_pop, ind_sol, motors)

        # score keeping
        gen_best = np.max(ind_sol)  # fitness of best individual
        best_indv = pop[ind_sol.index(gen_best)]  # solution of best individual
        best_gen_sol.append(best_indv)  # list of best individuals
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
            # randomizes worst individual
            movements.clear()
            change_index = ind_sol.index(np.min(ind_sol))
            changed_indv = pop[change_index]
            for k in range(len(changed_indv)):
                changed_indv[k] = uniform(bounds[k][0], bounds[k][1])
                movements.append(np.abs(motors[k].position - changed_indv[k]))
            max_move_index = movements.index(np.max(movements))
            update_velocity(motors, movements)
            for k in range(len(changed_indv)):
                if k == max_move_index:
                    st = motors[k].set(changed_indv[k])
                else:
                    motors[k].set(changed_indv[k])
            while not st.done:
                time.sleep(0.00001)
            ind_sol[change_index] = xs.channel1.rois.roi01.value.get()

    # Stop xspress3 acquisition
    xs.settings.acquire.put(0)

    # best solution overall should be last one
    x_best = best_gen_sol[-1]
    print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
    print('It took', v, 'generations')

    plot_index = np.arange(len(best_fitness))
    plt.figure()
    plt.plot(plot_index, best_fitness)
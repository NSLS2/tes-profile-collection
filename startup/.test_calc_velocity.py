import pytest


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



def test_calc_velocity():
    params_to_change = []
    params_to_change.append({'sample_stage_x': 55,
                             'sample_stage_y': 60,
                             'sample_stage_z': 15})


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

    calc_velocity(['a', 'b', 'c'], )
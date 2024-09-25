print(f"Loading {__file__!r} ...")

import bluesky.plan_stubs as bps

holder_on_sample_stage_status = 0
holder_empty_slot = 0


def High_throughput_XAS():
    pass
#[holder_type, holder_ID, holer_owner, holder_content] = sample_holders()





def load_holder(holder_type, holder_id):
    holder_type = str(holder_type)
    holder_id = str(holder_id)
    if home_check():
        yield from go_robot_parking()
        yield from go_SDD_parking()
        yield from go_sample_loading_area()
        yield from go_catch_holder(holder_type, holder_id)
        yield from go_load_holder()
        yield from go_robot_parking()
def unload_holder(holder_type, holder_id):
    holder_type = str(holder_type)
    holder_id = str(holder_id)
    if home_check():
        yield from go_SDD_parking()
        yield from go_sample_loading_area()
        yield from go_unload_holder()
        yield from go_return_holder(holder_type, holder_id)
        yield from go_robot_parking()




def scan_holder(holder_type, holder_owner, sample_list):




    if ready_to_scan():
        sample_position =sample_list[:,0]
        scan_type = sample_list[:, 4]
        element = sample_list[:,2]
        edge = sample_list[:,3]
        sample_name = sample_list[:,1]
        dwell_time = sample_list[:,7]
        num_scans = sample_list[:,8]

        for ii in range(sample_position.size):

            yield from bps.mv(sample_smart.x, sample_positions[holder_type][str(sample_position[ii])][0])
            absorption_edge = element_to_roi_smart[element[ii].lower()+"_"+edge[ii].lower()][2]
            scan_range = list(map(float, sample_list[:, 5][ii].strip('][').split(',')))
            E_scan_range = np.add(scan_range, absorption_edge)
            print("absorption edge = ", absorption_edge)
            print("scan range = ", E_scan_range)
            if scan_type[ii].lower() == "stepscan":


                if type(sample_list[ii, 6]) == int or type(sample_list[ii, 6]) == float:
                    step_size = [sample_list[ii, 6]]
                else:
                    step_size = list(map(float, sample_list[:, 6][ii].strip('][').split(',')))

                print("step size = ", step_size)

                yield from E_Step_Scan(scan_title = sample_name[ii], operator = holder_owner,
                                       element = element[ii].lower(), edge = edge[ii].lower(), detector = "xssmart",
                                       dwell_time=dwell_time[ii], E_sections = E_scan_range,
                                       step_size = step_size, num_scans = num_scans[ii])

            elif scan_type[ii].lower() == "flyscan":
                E_start = E_scan_range[0]
                E_stop = E_scan_range[1]
                step_size = sample_list[:, 6][ii]

                print("absorption edge = ", absorption_edge)
                print("scan range = ", E_start,E_stop)
                print("scan range = ", step_size)

                yield from E_fly_smart(scan_title= sample_name[ii], operator= holder_owner,
                                       element=element[ii], edge = edge[ii], start = E_start,
                                       stop=E_stop, step_size = step_size, num_scans = num_scans[ii], flyspeed=0.05)
            else:
                print("Undefined Scan Type")
                pass




    print("===========================saving data  " )
    yield from bps.sleep(5)


def auto_scan(holder_index = None):
    if input ("Please make sure the sample stage is EMPTY?") == "y":
        [holder_owners,holder_types, holder_lists, sample_lists]= sample_holders(holder_index)
        for ii in range(holder_lists.size):
            holder_type = holder_types[ii]
            holder_owner = holder_owners[ii]
            holder_id = holder_lists[ii]
            sample_list = sample_lists[ii]
            yield from load_holder(holder_type, holder_id)
            yield from go_scan()
            yield from scan_holder(holder_type, holder_owner, sample_list)
            yield from unload_holder(holder_type, holder_id)
            home_robot_smart("Y")
            yield from bps.mv(robot_smart.ry, 0)
            home_robot_smart("Ry")



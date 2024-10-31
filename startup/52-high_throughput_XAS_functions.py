print(f"Loading {__file__!r} ...")

import datetime
import os.path
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ophyd.status import WaitTimeoutError
from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


sample_x_loading = stage_positions["sample_loading"][0]
sample_y_loading = stage_positions["sample_loading"][1]
sample_z_loading = stage_positions["sample_loading"][2]
sample_ry_loading = stage_positions["sample_loading"][3]
sample_x = stage_positions["sample_scan"][0]
sample_y = stage_positions["sample_scan"][1]
sample_z = stage_positions["sample_scan"][2]
sample_ry = stage_positions["sample_scan"][3]
SDD_parking = stage_positions["SDD_smart_parking"]
SDD_scan = stage_positions["SDD_smart_scan"]
robot_x_parking = stage_positions["robot_parking"][0]
robot_y_parking = stage_positions["robot_parking"][1]
robot_z_parking = stage_positions["robot_parking"][2]
robot_ry_parking = stage_positions["robot_parking"][3]
robot_x_rotating = stage_positions["robot_rotating"][0]
robot_y_rotating = stage_positions["robot_rotating"][1]
robot_z_rotating = stage_positions["robot_rotating"][2]
robot_ry_rotating = stage_positions["robot_rotating"][3]
robot_x_loading = stage_positions["robot_loading"][0]
robot_y_loading = stage_positions["robot_loading"][1]
robot_z_loading = stage_positions["robot_loading"][2]
robot_ry_loading = stage_positions["robot_loading"][3]

# status of robot include
# parked
# moved
# loading
# catch 1x
# catch 2x
# ready to rotate

robot_status = ""


# status of sample stage includes
# ready to scan
# loading
stage_status = ""



# status of SDD includes
# parked
# parking
# sample loaded
# Empty
# ready to scan
SDD_status = ""



def home_check ():
    Homed = 1
    if robot_x_home.get() == 0:
        print("Robt X Axis is not homed")
        Homed = Homed * 0
    if robot_y_home.get() == 0:
        print("Robt Y Axis is not homed")
        Homed = Homed * 0
    if robot_z_home.get() == 0:
        print("Robt Z Axis is not homed")
        Homed = Homed * 0
    if robot_ry_home.get() == 0:
        print("Robt Ry Axis is not homed")
        Homed = Homed * 0
    if sample_x_home.get() == 0:
        print("Sample Stage X Axis is not homed")
        Homed = Homed * 0
    if sample_y_home.get() == 0:
        print("Sample Stage Y Axis is not homed")
        Homed = Homed * 0
    if sample_z_home.get() == 0:
        print("Sample Stage Z Axis is not homed")
        Homed = Homed * 0
    if sample_ry_home.get() == 0:
        print("Sample Stage Ry Axis is not homed")
        Homed = Homed * 0
    if SDD_smart_home.get == 0:
        print("Sample Stage Ry Axis is not homed")
        Homed = Homed * 0
    return Homed

# read the xls file and return sample holder informations
def sample_holders(holder_index = None):
    # read from xls file
    file_path = os.path.join(get_ipython().profile_dir.location, 'config/HighthroughputXAS.xls')
    data = np.array(pd.read_excel(file_path, sheet_name="ini", index_col=0).dropna())
    holder_owner = data [:,0]
    holder_type = data[:,1]
    holder_lists = []
    if holder_index == None:
        holder_lists = data[:,2]
    else:
        for ii in holder_index:
            holder_list = int(data[ii,2]);
            holder_lists.append(holder_list)
            #print(holder_lists)
        holder_lists = np.array(holder_lists)

    sample_lists = []
    for ii in holder_lists:
        sample_list = pd.read_excel(file_path, sheet_name=str(ii), index_col=0).dropna()
        sample_lists.append(sample_list)
    sample_lists = np.array(sample_lists)
    #print(holder_owner,holder_type,holder_lists,sample_list)
    return holder_owner, holder_type, holder_lists, sample_lists


# sample stage go to loading area
# SDD back to park first
def go_sample_loading_area():
    global stage_status, SDD_status

    if home_check():
        yield from bps.mv(SDD_smart.x, SDD_parking)
        yield from bps.mv(sample_smart.x, sample_x_loading, sample_smart.y, sample_y_loading, sample_smart.z, sample_z_loading, sample_smart.ry, sample_ry_loading)
        stage_status = "ready to load"
        print("Stage ready to load/unload")
        return True
    else:
        print("stage NOT homed")
        return False

def go_robot_parking():
    global robot_status
    position_robot_ry = robot_smart.ry.position

    if home_check():
        print ("parking")
        if abs(position_robot_ry + 90) <= 1 :
            robot_status = "moving"
            yield from bps.mv(robot_smart.x, robot_x_parking)
            yield from bps.mv(robot_smart.z, robot_z_parking)
            yield from bps.mv(robot_smart.y, robot_y_parking)
            yield from bps.mv(robot_smart.ry, robot_ry_parking)
            robot_status = "parked"
            print("Robot parked")
            return True
        elif abs(position_robot_ry + 180) <= 1:
            robot_status = "moving"
            yield from bps.mv(robot_smart.z, robot_z_parking)
            yield from bps.mv(robot_smart.x, robot_x_parking)
            yield from bps.mv(robot_smart.y, robot_y_parking)
            yield from bps.mv(robot_smart.ry, robot_ry_parking)
            robot_status = "parked"
            print("Robot parked")
            return True
        else:
            print("Please manually adjust the robot Ry")
            return False
    else:
        print("stage NOT homed")

# catch and lift the holder
# done at 9/4 15:53
def go_catch_holder (holder_type, holder_Index):
    global robot_status
    # 1x is holder inboard
    # 2x are the holder outboard
    holder_x = holder_positions[holder_type][holder_Index][0]
    holder_y = holder_positions[holder_type][holder_Index][1]
    holder_z = holder_positions[holder_type][holder_Index][2]
    holder_ry = holder_positions[holder_type][holder_Index][3]

    # starting from parking position only
    if home_check():
        print ("going to catch holder")
        if robot_smart_status() == "parked":
            yield from bps.mv(robot_smart.ry, holder_ry)
            if int(holder_Index) < 20:
                robot_status = "moved"
                yield from bps.mv(robot_smart.y, holder_y)
                yield from bps.mv(robot_smart.z, holder_z)
                yield from bps.mv(robot_smart.x, holder_x)
                yield from bps.mv(robot_smart.y, holder_y+20)
                robot_status = "catch 1x"
                print("1x caugth")
                return robot_status

            elif int(holder_Index) > 20:
                robot_status = "moved"
                yield from bps.mv(robot_smart.y, holder_y)
                yield from bps.mv(robot_smart.x, holder_x)
                yield from bps.mv(robot_smart.z, holder_z)
                yield from bps.mv(robot_smart.y, holder_y+20)
                robot_status = "catch 2x"
                print("2x caught")
                return robot_status
        else:
            yield from go_robot_parking()
    else:
        print("stage NOT homed")

# 1x inboard
# 2x outboard
# load the holder to the sample stage after the holder lifted by the robot
# done 9/4 16:51
def go_load_holder():
    global robot_status, stage_status, SDD_status

    if home_check():
        print("Going to load holder")
        if stage_smart_status() == "ready to load" and SDD_smart_status() == "parked":

            if robot_status == "catch 1x":
                yield from bps.mv(robot_smart.x, robot_x_rotating)
                yield from bps.mv(robot_smart.z, robot_z_rotating)
                yield from bps.mv(robot_smart.ry, robot_ry_loading)
                yield from bps.mv(robot_smart.x, robot_x_loading)
                yield from bps.mv(robot_smart.z, robot_z_loading)
                yield from bps.mv(robot_smart.y, robot_y_loading)
                yield from go_robot_parking()
                stage_status = "sample loaded"
                print("sample loaded")
                return True
            elif robot_status == "catch 2x":
                yield from bps.mv(robot_smart.z, robot_z_rotating)
                yield from bps.mv(robot_smart.x, robot_x_rotating)
                yield from bps.mv(robot_smart.y, robot_y_rotating)
                yield from bps.mv(robot_smart.ry, robot_ry_rotating)
                yield from bps.mv(robot_smart.ry, robot_ry_loading)
                yield from bps.mv(robot_smart.x, robot_x_loading)
                yield from bps.mv(robot_smart.z, robot_z_loading)
                yield from bps.mv(robot_smart.y, robot_y_loading)
                yield from go_robot_parking()
                stage_status = "sample loaded"
                return True
                print("sample loaded")
            else:
                print("Sample holder not caught")
                return False
        else:
            print("NOT ready to load sample holder yet")
            return False
    else:
        print("stage NOT homed")
        return False


def go_unload_holder():
    global  robot_status, SDD_status, stage_status
    if home_check():
        if robot_smart_status() == "parked" or robot_smart_status() == "ready to rotate" and stage_smart_status() == "ready to load" and SDD_smart_status() == "parked":
            robot_status = "moved"
            yield from bps.mv(robot_smart.x, robot_x_rotating, robot_smart.y, robot_y_rotating, robot_smart.z, robot_z_rotating, robot_smart.ry, robot_ry_loading)
            yield from bps.mv(robot_smart.y, robot_y_loading)
            yield from bps.mv(robot_smart.x, robot_x_loading)
            yield from bps.mv(robot_smart.z, robot_z_loading)
            yield from bps.mv(robot_smart.y, robot_y_loading + 20)
            robot_status = "unloading"
            print (robot_status)
        else:
            yield from go_robot_parking()
            yield from go_sample_loading_area()
            robot_status = "moved"
            yield from bps.mv(robot_smart.x, robot_x_rotating, robot_smart.y, robot_y_rotating, robot_smart.z, robot_z_rotating, robot_smart.ry, robot_ry_loading)
            yield from bps.mv(robot_smart.y, robot_y_loading)
            yield from bps.mv(robot_smart.x, robot_x_loading)
            yield from bps.mv(robot_smart.z, robot_z_loading)
            yield from bps.mv(robot_smart.y, robot_y_loading + 20)
            robot_status = "unloading"
            print(robot_status)
    else:
        print("stage NOT homed")



def go_return_holder(holder_type, holder_Index):
    holder_x = holder_positions[holder_type][holder_Index][0]
    holder_y = holder_positions[holder_type][holder_Index][1]
    holder_z = holder_positions[holder_type][holder_Index][2]
    holder_ry = holder_positions[holder_type][holder_Index][3]

    if robot_smart_status() == "unloading":
        yield from bps.mv(robot_smart.z, robot_z_rotating)
        yield from bps.mv(robot_smart.x, robot_x_rotating)
        yield from bps.mv(robot_smart.ry, holder_ry)

        if int(holder_Index) < 20:
            yield from bps.mv(robot_smart.z, holder_z)
            yield from bps.mv(robot_smart.x, holder_x)
            yield from bps.mv(robot_smart.y, holder_y)
            yield from go_robot_parking()
            return True
        elif int(holder_Index) > 20:
            yield from bps.mv(robot_smart.x, holder_x)
            yield from bps.mv(robot_smart.z, holder_z)
            yield from bps.mv(robot_smart.y, holder_y)
            yield from go_robot_parking()
            return True
        else:
            print("Holder index is not in the list")
            return False
    else:
        if input("Are you sure want to return the holder? y/n   ") == "y":
            yield from go_robot_parking()
            yield from bps.mv(robot_smart.ry, holder_ry)
            yield from bps.mv(robot_smart.y, 40)
            if int(holder_Index) < 20:

                yield from bps.mv(robot_smart.z, holder_z)
                yield from bps.mv(robot_smart.x, holder_x)
                yield from bps.mv(robot_smart.y, holder_y)
                yield from go_robot_parking()
                return True
            elif int(holder_Index) > 20:
                yield from bps.mv(robot_smart.x, holder_x)
                yield from bps.mv(robot_smart.z, holder_z)
                yield from bps.mv(robot_smart.y, holder_y)
                yield from go_robot_parking()
                return True
            else:
                print("Holder index is not in the list")
                return False
        else:
            print ("Please reset the system manually")




def go_SDD_parking():
    global SDD_status
    if home_check():
        SDD_status = "moving"
        yield from bps.mv(SDD_smart.x, SDD_parking)
        SDD_status = "parked"
        return SDD_status
    else:
        print("stage NOT homed")

def go_SDD_Scaning():
    global SDD_status
    if home_check():
        SDD_status = "moving"
        yield from bps.mv(SDD_smart.x, SDD_parking)
        SDD_status = "ready to scan"
        return SDD_status
    else:
        print("stage NOT homed")

def go_scan():
    global robot_status, stage_status, SDD_status
    if home_check():
        yield from bps.mv(sample_smart.x, sample_x, sample_smart.y, sample_y, sample_smart.z, sample_z, sample_smart.ry, sample_ry, SDD_smart.x, SDD_scan)

    else:
        print("stage NOT homed")

def ready_to_scan():
    if home_check():
        if stage_smart_status() == "Not ready to load" and SDD_smart_status() != False and robot_smart_status() == "parked":
            return True
        else:
            return False

def stage_smart_status():
    global stage_status
    if home_check():
        if abs(sample_smart.x.position - sample_x_loading) < 0.01 and abs(sample_smart.y.position - sample_y_loading) < 0.01 and abs(
                sample_smart.z.position - sample_z_loading) < 0.01 and abs(sample_smart.ry.position - sample_ry_loading) < 0.01 and SDD_smart_status()=="parked" and robot_smart_status() == "parked":
            stage_status = "ready to load"

        else:
            stage_status = "Not ready to load"

        return stage_status
    else:
        print("stage NOT homed")
        stage_status = "take no reaction"
    return stage_status
def robot_smart_status():
    global robot_status
    if home_check():
        if abs(robot_smart.x.position - robot_x_parking) < 0.01 and abs(robot_smart.y.position - robot_y_parking) < 0.01 and abs(robot_smart.z.position - robot_z_parking) < 0.01 and abs(robot_smart.ry.position - robot_ry_parking) < 0.01:
            robot_status = "parked"
        elif abs(robot_smart.x.position - robot_x_rotating) <0.01 and abs(robot_smart.y.position - robot_y_rotating)<0.01 and abs(robot_smart.z.position - robot_z_rotating)<0.01:
            robot_status = "ready to rotate"
        elif abs(robot_smart.x.position - robot_x_loading) <0.01 and  abs(robot_smart.z.position - robot_z_loading)<0.01 and abs(robot_smart.ry.position - robot_ry_loading)<0.01:
            robot_status = "unloading"
        else:
            pass
    else:
        print("stage NOT homed")
    return robot_status

def SDD_smart_status():
    if home_check():
        if abs(SDD_smart.x.position - SDD_parking) <0.1:
            return "parked"

        else:
            return "ok"

    else:
        print("stage NOT homed")


def home_robot_smart(axis = "Y"):
    home_robot_smart = EpicsSignal("XF:08BMC-ES:SE{SmplM:1-Ax:"+axis+"}Start:Home-Cmd", name = "home_robot_smart")
    home_robot_smart.put(1)





'''

def ready_to_scan ():
    #parking
    #detector in position
    #sample holder in position
    #beam on
    #detector readout reasonable
    if home_check():
        if robot_smart_status() == "parked" and sample_stage_status() == "ready to scan":
            return True
        else:
            return False







    
def robot_smart_status():
    global robot_status
    if home_check():
        if abs(robot_smart.x.position - robot_x_parking) < 0.01 and abs(robot_smart.y.position - robot_y_parking) < 0.01 and abs(robot_smart.z.position - robot_z_parking) < 0.01 and abs(robot_smart.ry.position - robot_ry_parking) < 0.01:
            robot_status = "parked"
        elif abs(robot_smart.x.position - robot_x_rotating) <0.01 and abs(robot_smart.y.position - robot_y_rotating)<0.01 and abs(robot_smart.z.position - robot_z_rotating)<0.01:
            robot_status = "ready to rotate"
        elif abs(robot_smart.x.position - robot_x_loading) <0.01 and abs(robot_smart.y.position - robot_y_loading)<0.01 and abs(robot_smart.z.position - robot_z_loading)<0.01 and abs(robot_smart.ry.position - robot_ry_loading):
            robot_status = "loading"
    else:
        print("stage NOT homed")
        robot_status = "take no action"
    return robot_status

def SDD_smart_status():
    if home_check():
        if abs(SDD_smart.x.position - SDD_parking) <0.1:
            return "parked"
        elif abs(SDD_smart.x.position - SDD_scan) <0.1:
            return "scanning"
        else:
            return "parking"
    else:
        print("stage NOT homed")
'''

"""
def go_robot_loading():
    position_robot_ry = robot_smart.ry.position
    if home_check():
        if robot_smart_status() == "parked":
            yield from bps.mv(robot_smart.ry, robot_ry_rotating)

        elif abs(position_robot_ry + 180) <= 1:
            yield from bps.mv(robot_smart.z, robot_z_parking)
            yield from bps.mv(robot_smart.x, robot_x_parking)
            yield from bps.mv(robot_smart.y, robot_y_parking)
            yield from bps.mv(robot_smart.ry, robot_ry_parking)
            return True
        else:
            print("Something wrong")
            return False
    else:
        print("stage NOT homed")


def robot_parked ():
    while home_check():
        if abs(robot_smart.x - robot_x_parking) < 0.01 and abs(robot_smart.y - robot_y_parking) < 0.01 and abs(robot_smart.z - robot_z_parking) < 0.01 and abs(robot_smart.ry - robot_ry_parking) < 0.01:
            return True
        else:
            return False
"""

import pprint
from bluesky.plans import list_scan
from bluesky.plans import grid_scan
from bluesky.plan_stubs import abs_set

from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from event_model import RunRouter
import numpy as np
import pandas as pd
from bluesky.callbacks.mpl_plotting import LivePlot

'''
# TODO could also use check_value, but like the better error message here?
def _validate_motor_limits(motor, start, stop, k):
    # blow up on inverted values
    assert start < stop, (
        f"start ({start}) must be smaller than " f"stop ({stop}) for {k}"
    )
    limits = motor.limits
    if any(not (limits[0] < v < limits[1]) for v in (start, stop)):
        raise LimitError(
            f"your requested {k} values are out of limits for "
            "the motor "
            f"{limits[0]} < ({start}, {stop}) < {limits[1]}"
        )


def _get_v_with_dflt(sig, dflt):
    ret = yield from bps.read(sig)
    return ret[sig.name]["value"] if ret is not None else dflt


x_centers = Signal(value=[], name="x_centers", kind="normal")
x_centers.tolerance = 1e-15
y_centers = Signal(value=[], name="y_centers", kind="normal")
y_centers.tolerance = 1e-15
z_centers = Signal(value=[], name="z_centers", kind="normal")
z_centers.tolerance = 1e-15

'''

def _get_v_with_dflt(sig, dflt):
    ret = yield from bps.read(sig)
    return ret[sig.name]["value"] if ret is not None else dflt




#@bpp.baseline_decorator([mono, xy_stage])
def E_Step_Scan(scan_title, *, operator, element, dwell_time=3, E_sections, step_size, num_scans, xspress3):
#def E_Step_Scan(dwell_time,*, scan_title = "abc",E_sections = [2700, 2800, 2900, 3200], step_size = [4, 1, 2], num_scans=2, element = 's'):

    e_back = yield from _get_v_with_dflt(mono.e_back, 1977.04)
    energy_cal = yield from _get_v_with_dflt(mono.cal, 0.40118)
    print(e_back,energy_cal)
    def _energy_to_linear(energy):
        energy = np.asarray(energy)
        return 28.2474 + 35.02333 * np.tan(
            np.pi / 2 - 2 * np.arcsin(e_back / energy) + np.deg2rad(energy_cal)
        )

    #for v in ["p1600=0", "p1607=4", "p1601=5", "p1602 = 2", "p1600=1"]:
        #yield from bps.mv(dtt, v)
        #yield from bps.sleep(0.1)
    roi = rois(element)
#    yield from bps.mv(xs.channel1.rois.roi01.bin_low, roi[0],
#                  xs.channel1.rois.roi01.bin_high, roi[1])
#    yield from bps.sleep(0.1)
#    xs.channel1.rois.roi01.bin_low.set(roi[0])
#    xs.channel1.rois.roi01.bin_high.set(roi[1])
    E_sections = np.array(E_sections)
    step_size = np.array(step_size)

    ept = []
    for ii in range(step_size.shape[0]):
        ept = ept[0:-1]
        ept = np.append(ept, np.linspace(E_sections[ii], E_sections[ii+1], int((E_sections[ii+1] - E_sections[ii])/step_size[ii])+1))
#        print(ept)
    sclr.set_mode("counting")
    yield from bps.mv(xspress3.external_trig, False) # xs triger mode false means internal trigger
    yield from bps.mv(xspress3.cam.num_images, 1)
    yield from bps.mv(sclr.cnts.preset_time, dwell_time,
                      xspress3.cam.acquire_time, dwell_time)#setting dwell time
    ept_linear =  _energy_to_linear(ept)
    #yield from bps.mv(sclr.set_mode,"counting")
    yield from bps.mv(mono.linear.velocity, 0.2)
    #yield from bps.sleep(0.1)
    #@bpp.monitor_during_decorator([xs.channel1.rois.roi01.value])
    #@bpp.baseline_decorator([mono, xy_stage])
    # TODO put in other meta data
    def scan_once():
 #       l_start = _energy_to_linear(ept[0])

 #       yield from bps.mv(mono.linear, l_start)



        yield from bps.checkpoint()
        return (yield from list_scan(
            [sclr, xspress3],
            mono.linear,
            ept_linear,
            md={
                "scan_title": scan_title,
                "operator": operator,
                "element": element,
                "user_input": {
                    "element": element,
                    "E_sections": E_sections,
                    "dwell_time": dwell_time,
                    "step_size": step_size,
                },
                "derived_input": {
                    "x":xy_stage.x.position,
                    "y":xy_stage.y.position,
                    "z":xy_stage.z.position,
                   # "Monochromator": Si(111)
                },
            }))
        #, LivePlot('sclr', 'mono_linear'))
        #yield from bps.mv(mono.linear.velocity, 0.1)


    for scan_iter in range(int(num_scans)):



        yield from scan_once()
        yield from export_E_step(-1,scan_iter)


"""
    print("Waiting for files... ...")
    yield from bps.sleep(15)
    for scan_n in range(1, num_scans+1):
        artifacts = e_step_export(db[-1*scan_n])
        pprint.pprint(artifacts)
"""


# np.fromstring('1, 2', dtype=int, sep=',')




import datetime
import os.path
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from bluesky.callbacks import LiveGrid

# Add extra devices to the baseline.
# Don't use the baseline decorator because it can create conflicts.
sd.baseline.extend([mono, xy_stage])

# Testing VI with Yonghua


# TODO could also use check_value, but like the better error message here?
def _validate_motor_limits(motor, start, stop, k):
    # blow up on inverted values
    assert start < stop, (
        f"start ({start}) must be smaller than " f"stop ({stop}) for {k}"
    )
    limits = motor.limits
    if any(not (limits[0] < v < limits[1]) for v in (start, stop)):
        raise LimitError(
            f"your requested {k} values are out of limits for "
            "the motor "
            f"{limits[0]} < ({start}, {stop}) < {limits[1]}"
        )


def _get_v_with_dflt(sig, dflt):
    ret = yield from bps.read(sig)
    return ret[sig.name]["value"] if ret is not None else dflt


x_centers = Signal(value=[], name="x_centers", kind="normal")
x_centers.tolerance = 1e-15
y_centers = Signal(value=[], name="y_centers", kind="normal")
y_centers.tolerance = 1e-15
z_centers = Signal(value=[], name="z_centers", kind="normal")
z_centers.tolerance = 1e-15


def xy_step(
        scan_title,
        *,
        beamline_operator,
        dwell_time,
        xstart,
        xstop,
        xstep_size,
        ystart,
        ystop,
        ystep_size=None,
        xspress3=None,
):
    """Do a x-y fly scan.

    The x-motor is the 'fast' direction.

    Parameters
    ----------
    scan_title : str
       A name for the scan.

    beamline_operator : str
       The individual responsible for this scan. Appears in output directory path.

    dwell_time : float
       Target time is s on each pixel

    xstart, xstop : float
       The start and stop values in the fast direction in mm

    xstep_size :
        xstep_size is step of x movement

    ystart, ystop : float
       The start and stop values in the slow direction in mm

    ystep_size :
        ystep_size use xstep_size if it isn't passed in

    scan_title : str
       Title of scan, required.
    """
    if xspress3 != None:
        xspress3 = xs
    xy_fly_stage = xy_stage
    _validate_motor_limits(xy_fly_stage.x, xstart, xstop, "x")
    _validate_motor_limits(xy_fly_stage.y, ystart, ystop, "y")
    ystep_size = ystep_size if ystep_size is not None else xstep_size
    assert dwell_time > 0, f"dwell_time ({dwell_time}) must be more than 0"
    assert xstep_size > 0, f"xstep_size ({xstep_size}) must be more than 0"
    assert ystep_size > 0, f"ystep_size ({ystep_size}) must be more than 0"
    ret = yield from bps.read(xy_fly_stage.x.mres)  # (in mm)
    #xmres = ret[xy_fly_stage.x.mres.name]["value"] if ret is not None else 0.0003125
    xmres = ret[xy_fly_stage.x.mres.name]["value"] if ret is not None else 0.0002
    ret = yield from bps.read(xy_fly_stage.y.mres)  # (in mm)
    ymres = ret[xy_fly_stage.y.mres.name]["value"] if ret is not None else 0.0002

    # to reach 0.4um step size
    prescale = int(np.floor((xstep_size / (5 * xmres))))
    #prescale = int(np.floor((xstep_size / (2*xmres))))
    a_xstep_size = prescale * (5*xmres)
    #a_xstep_size = xstep_size;
    a_ystep_size = int(np.floor((ystep_size / (ymres)))) * ymres

    num_xpixels = int(np.floor((xstop - xstart) / a_xstep_size))
    num_ypixels = int(np.floor((ystop - ystart) / a_ystep_size))

    yield from bps.mv(
        x_centers, a_xstep_size / 2 + xstart + np.arange(num_xpixels) * a_xstep_size
    )

    # SRX original roi_key = getattr(xs.channel1.rois, roi_name).value.name

    #    roi_livegrid_key = xs.channel1.rois.roi01.value.name
    #fig = plt.figure("xs")
    #fig.clf()
    #roi_livegrid = LiveGrid(
    #    (num_ypixels + 1, num_xpixels + 1),
    #    roi_livegrid_key,
    #    clim=None,
    #    cmap="inferno",
    #    xlabel="x (mm)",
    #    ylabel="y (mm)",
    #    extent=[xstart, xstop, ystart, ystop],
    #    x_positive="right",
    #    y_positive="down",
    #    ax=fig.gca(),
    #)

    flyspeed = 1  # this is in mm/s

    try:
        xy_fly_stage.x.velocity.check_value(flyspeed)
    except LimitError as e:
        raise LimitError(
            f"You requested a range of {xstop - xstart} with "
            f"{num_xpixels} pixels and a dwell time of "
            f"{dwell_time}.  This requires a "
            f"motor velocity of {flyspeed} which "
            "is out of range."
        )


    # TODO make this a message?
    sclr.set_mode("counting")
    # poke the struck settings
    yield from bps.mv(sclr.mcas.prescale, prescale)
    yield from bps.mv(sclr.mcas.nuse, num_xpixels)
    if xspress3 is not None:
        yield from bps.mv(xs.external_trig, False)
        yield from bps.mv(xspress3.total_points, num_xpixels)
        yield from bps.mv(xspress3.hdf5.num_capture, num_xpixels)
        yield from bps.mv(xspress3.settings.num_images, num_xpixels)

    yield from bps.mv(xy_fly_stage.x, xstart, xy_fly_stage.y, ystart)

    yield from grid_scan([sclr, xs], xy_fly_stage.x, xstart,xstop,num_xpixels,
                         xy_fly_stage.y, ystart,ystop,num_ypixels)

    # save the start document to a file for the benefit of the user
    #export_xy_fly()


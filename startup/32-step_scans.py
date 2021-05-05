import pprint

from bluesky.plans import list_scan
from bluesky.plan_stubs import abs_set

from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from event_model import RunRouter
import numpy as np
import pandas as pd

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


def _linear_to_energy(linear):
    linear = np.asarray(linear)
    return e_back / np.sin(
        np.deg2rad(45)
        + 0.5 * np.arctan((28.2474 - linear) / 35.02333)
        + np.deg2rad(energy_cal) / 2
    )


def _energy_to_linear(energy):
    energy = np.asarray(energy)
    return 28.2474 + 35.02333 * np.tan(
        np.pi / 2 - 2 * np.arcsin(e_back / energy) + np.deg2rad(energy_cal)
    )
'''

#@bpp.baseline_decorator([mono, xy_stage])
def E_Step_Scan(scan_title, *, operator, element, dwell_time=3, E_sections, step_size, num_scans, xspress3=None):
#def E_Step_Scan(dwell_time,*, scan_title = "abc",E_sections = [2700, 2800, 2900, 3200], step_size = [4, 1, 2], num_scans=2, element = 's'):


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
        ept = np.append(ept, np.linspace(E_sections[ii], E_sections[ii+1], np.int((E_sections[ii+1] - E_sections[ii])/step_size[ii])+1))
#        print(ept)
    sclr.set_mode("counting")
    yield from bps.mv(xs.external_trig, False)
    yield from bps.mv(sclr.cnts.preset_time, dwell_time,
                      xs.settings.acquire_time, dwell_time)

    #yield from bps.mv(sclr.set_mode,"counting")

    #yield from bps.sleep(0.1)
    #@bpp.monitor_during_decorator([xs.channel1.rois.roi01.value])
    #@bpp.baseline_decorator([mono, xy_stage])
    # TODO put in other meta data
    def scan_once():
        yield from bps.mv(mono.linear.velocity, 0.5)
        return (yield from list_scan(
            [sclr, xs],
            mono.energy,
            ept,
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
            }

        ))
        yield from bps.mv(mono.linear.velocity, 0.1)


    for scan_iter in range(num_scans):
        yield from scan_once()
        export_E_step(-1,scan_iter)


"""
    print("Waiting for files... ...")
    yield from bps.sleep(15)
    for scan_n in range(1, num_scans+1):
        artifacts = e_step_export(db[-1*scan_n])
        pprint.pprint(artifacts)
"""


# np.fromstring('1, 2', dtype=int, sep=',')

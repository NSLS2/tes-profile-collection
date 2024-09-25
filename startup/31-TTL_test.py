print(f"Loading {__file__!r} ...")

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


def TTL_test(
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

    prescale = int(np.floor((xstep_size / (5 * xmres))))
    a_xstep_size = prescale * (5 * xmres)

    a_ystep_size = int(np.floor((ystep_size / (ymres)))) * ymres

    num_xpixels = int(np.floor((xstop - xstart) / a_xstep_size))
    num_ypixels = int(np.floor((ystop - ystart) / a_ystep_size))

    yield from bps.mv(
        x_centers, a_xstep_size / 2 + xstart + np.arange(num_xpixels) * a_xstep_size
    )

    # SRX original roi_key = getattr(xs.channel1.rois, roi_name).value.name
    roi_livegrid_key = xs.channel1.rois.roi01.value.name
    fig = plt.figure("xs")
    fig.clf()
    roi_livegrid = LiveGrid(
        (num_ypixels + 1, num_xpixels + 1),
        roi_livegrid_key,
        clim=None,
        cmap="inferno",
        xlabel="x (mm)",
        ylabel="y (mm)",
        extent=[xstart, xstop, ystart, ystop],
        x_positive="right",
        y_positive="down",
        ax=fig.gca(),
    )

    flyspeed = a_xstep_size / dwell_time  # this is in mm/s

    try:
        xy_fly_stage.x.velocity.check_value(flyspeed)
    except LimitError as e:
        raise LimitError(
            f"You requested a range of {xstop - xstart} with "
            f"{num_xpixels} pixels and a dwell time of "
            f"{dwell_time}.  This requires a "
            f"motor velocity of {flyspeed} which "
            "is out of range."
        ) from e

    # set up delta-tau trigger to fast motor
    for v in ["p1600=0", "p1607=1", "p1600=1"]:
        yield from bps.mv(dtt, v)
        yield from bps.sleep(0.1)

    # TODO make this a message?
    sclr.set_mode("flying")
    # poke the struck settings
    yield from bps.mv(sclr.mcas.prescale, prescale)
    yield from bps.mv(sclr.mcas.nuse, num_xpixels)

    @bpp.reset_positions_decorator([xy_fly_stage.x, xy_fly_stage.y])
    @bpp.stage_decorator([sclr])
    #@bpp.baseline_decorator([mono, xy_fly_stage])
    # TODO put is other meta data
    @bpp.run_decorator(
        md={
            "scan_title": scan_title,
            "operator": beamline_operator,
            "user_input": {
                "dwell_time": dwell_time,
                "xstart": xstart,
                "xstop": xstop,
                "xstep_size": xstep_size,
                "ystart": ystart,
                "ystep_size": ystep_size,
            },
            "derived_input": {
                "actual_ystep_size": a_ystep_size,
                "actual_xstep_size": a_xstep_size,
                "fly_velocity": flyspeed,
                "xpixels": num_xpixels,
                "ypixels": num_ypixels,
                "prescale": prescale,
            },
        }
    )
    def fly_body():

        yield from bps.mv(xy_fly_stage.x, xstart, xy_fly_stage.y, ystart)

        def fly_row():
            # go to start of row
            target_y = ystart + y * a_ystep_size
            yield from bps.mv(xy_fly_stage.x, xstart, xy_fly_stage.y, target_y)
            yield from bps.mv(
                y_centers, np.ones(num_xpixels) * target_y
            )  # set the fly speed

            ret = yield from bps.read(xy_fly_stage.z.user_readback)  # (in mm)
            zpos = (
                ret[xy_fly_stage.z.user_readback.name]["value"]
                if ret is not None
                else 0
            )
            yield from bps.mov(z_centers, np.ones(num_xpixels) * zpos)

            yield from bps.mv(xy_fly_stage.x.velocity, flyspeed)

            yield from bps.trigger_and_read([xy_fly_stage], name="row_ends")

            for v in ["p1600=0", "p1600=1"]:
                yield from bps.mv(dtt, v)
                yield from bps.sleep(0.1)

            # arm the struck
            yield from bps.trigger(sclr, group=f"fly_row_{y}")
            # maybe start the xspress3
           # fly the motor
            yield from bps.abs_set(
                xy_fly_stage.x, xstop + a_xstep_size, group=f"fly_row_{y}"
            )
            yield from bps.wait(group=f"fly_row_{y}")

            yield from bps.trigger_and_read([xy_fly_stage], name="row_ends")
            yield from bps.mv(xy_fly_stage.x.velocity, 5.0)
            yield from bps.sleep(0.1)
            # read and save the struck
            yield from bps.create(name="primary")
            #
            yield from bps.read(sclr)
            yield from bps.read(mono)
            yield from bps.read(x_centers)
            yield from bps.read(y_centers)
            yield from bps.read(z_centers)
            yield from bps.read(xy_fly_stage.y)
            yield from bps.read(xy_fly_stage.z)
            # and maybe the xspress3
            yield from bps.save()

        for y in range(num_ypixels):

            yield from fly_row()

    yield from fly_body()

from ophyd.utils import LimitError
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np

xy_fly_stage = None
dtt = None
sclr = None
energy = None

# TODO could also use check_value, but like the better error message here?
def _validate_motor_limits(motor, start, stop, k):
    limits = motor.limits
    if any(not (limits[0] < v < limits[1]) for v in (start, stop)):
        raise LimitError(f"your requested {k} values are out of limits for the motor "
                         f"{limits[0]} < ({start}, {stop}) < {limits[1]}")


def xy_fly(dwell_time,
           xstart, xstop, num_xpixels,
           ystart, ystop, num_ypixels=None,
           *, scan_title):
    """Do a x-y fly scan.

    The x-motor is the 'fast' direction.

    Parameters
    ----------
    dwell_time : float
       Target time is ms on each pixel

    xstart, xstop : float
       The start and stop values in the fast direction in mm

    num_xpixels : int
       The number of pixels to collect in the fast direction

    ystart, ystop : float
       The start and stop values in the slow direction in mm

    num_ypixels: float, optional
       Number of pixels in the y direction.  If not given, defaults to
       the same as **num_xpixels**.

    scan_title : str
       Title of scan, required.
    """
    _validate_motor_limits(xy_fly_stage.x, xstart, xstop, 'x')
    _validate_motor_limits(xy_fly_stage.y, ystart, ystop, 'y')

    num_ypixels = num_ypixels if num_ypixels is not None else num_xpixels

    xstep_size = (xstop - xstart) / num_xpixels
    ystep_size = (ystop - ystart) / num_ypixels

    flyspeed = abs(xstop - xstart) / dwell_time  # this is in mm/ms == m/s

    try:
        xy_fly_stage.x.velocity.check_value(flyspeed)
    except LimitError as e:
        raise LimitError(f'You requested a range of {xstop - xstart} with '
                         f'{num_xpixels} pixels and a dwell time of '
                         f'{dwell_time}.  This requires a '
                         f'motor velocity of {flyspeed} which '
                         'is out of range.') from e

    # set up delta-tau trigger to fast motor
    for v in ['p1600=0', 'p1607=1', 'p1601=0']:
        yield from bps.mv(dtt, v)
        yield from bps.sleep(0.01)

    yield from bps.mv(xy_fly_stage.x, xstart,
                      xy_fly_stage.y, ystart)

    # TODO make this a message?
    sclr.set_mode('flying')

    # get the motor resolution
    ret = yield from bps.read(xy_fly_stage.x.mres)
    mres = (ret[xy_fly_stage.x.mres['name']]
            if ret is not None else np.nan)

    # poke the struck settings
    prescale = xstep_size * mres
    yield from bps.mv(sclr.prescale, prescale)
    yield from bps.mv(sclr.nuse, num_xpixels)

    # TODO check wrapper order
    @bpp.stage_decorator([sclr])
    @bpp.baseline_decorator([energy])
    # TODO put is other meta data
    @bpp.run_wrapper(md={'scan_title': scan_title})
    def fly_body():
        for y in range(num_ypixels):
            # go to start of row
            yield from bps.mv(xy_fly_stage.x, xstart,
                              xy_fly_stage.y, ystart + y*ystep_size)

            # set the fly speed
            yield from bps.mv(xy_fly_stage.x.velocity, flyspeed)

            for v in ['p1600=0', 'p1601=0']:
                yield from bps.mv(dtt, v)
                yield from bps.sleep(0.01)

            # arm the struck
            yield from bps.trigger(sclr, group=f'fly_row_{y}')
            # fly the motor
            yield from bps.abs_set(xy_fly_stage.x, xstop, group=f'fly_row_{y}')
            yield from bps.wait(group=f'fly_row_{y}')
            # read and save the struck
            yield from bps.create(name='primary')
            yield from bps.read(sclr)
            yield from bps.read(xy_fly_stage.y)
            yield from bps.save()

            yield from bps.mv(xy_fly_stage.x.velocity, 5.0)

    yield from fly_body()

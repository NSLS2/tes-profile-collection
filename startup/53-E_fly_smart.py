
print(f"Loading {__file__!r} ...")


import numpy as np
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp



E_centers = Signal(value=[], name="E_centers", kind="normal")
E_centers.tolerance = 1e-15

def validate_motor_limits(motor, start, stop, k):
    # blow up on inverted values
 #   assert start < stop, (
 #       f"start ({start}) must be smaller than " f"stop ({stop}) for {k}"
 #   )
    limits = motor.limits
    if any(not (limits[0] < v < limits[1]) for v in (start, stop)):
        raise LimitError(
            f"your requested {k} values are out of limits for "
            "the motor "
            f"{limits[0]} < ({start}, {stop}) < {limits[1]}"
        )



def E_fly_smart(
        scan_title, *, operator, element, edge, start, stop, step_size, num_scans, flyspeed=0.05, xspress3=None
):
    if xspress3 == None:
        xspress3 = xssmart

    xspress3.fluor.kind = Kind.normal
    
    validate_motor_limits(mono.energy, start, stop, "E")
    assert step_size > 0, f"step_size ({step_size}) must be more than 0"
    assert num_scans > 0, f"num_scans ({num_scans}) must be more than 0"

    e_back = yield from _get_v_with_dflt(mono.e_back, 1977.04)
    energy_cal = yield from _get_v_with_dflt(mono.cal, 0.40118)
#    roi = rois(element)
#    yield from bps.mv(xs.channel1.rois.roi01.bin_low, roi[0])
#    yield from bps.sleep(0.5)
#    yield from bps.mv(xs.channel1.rois.roi01.bin_high, roi[1])
#    yield from bps.sleep(0.5)

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

    # get limits in linear parameters
    l_start, l_stop = _energy_to_linear([start, stop])
    l_step_size = np.diff(_energy_to_linear([start, start + step_size]))

    # scale to match motor resolution
    lmres = yield from _get_v_with_dflt(mono.linear.mres, 0.0001666)

    prescale = int(np.floor((l_step_size / (5 * lmres))))
    a_l_step_size = prescale * (5 * lmres)

    num_pixels = int(np.floor((l_stop - l_start) / a_l_step_size))
    #For from high to low
    num_pixels = abs(num_pixels)
    print(f"l_start={l_start} l_stop={l_stop} a_l_step_size={a_l_step_size}")
    print(f"=========== num_pixels={num_pixels} ==============")

    bin_edges = _linear_to_energy(l_start + a_l_step_size * np.arange(num_pixels + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    yield from bps.mv(E_centers, bin_centers)

    # The flyspeed is set by Paul by edict
    # flyspeed = 0.05

    # set up delta-tau trigger to fast motor
    for v in ["p1600=0", "p1607=4", "p1601=5", "p1602 = 2", "p1604 = 0", "p1600=1"]:
        yield from bps.mv(dtt, v)
        yield from bps.sleep(0.1)

    # TODO make this a message?
    sclr.set_mode("flying")

    # SRX original roi_key = getattr(xs.channel1.rois, roi_name).value.name

    # roi_livegrid_key = xs.channel1.rois.roi01.value.name
    # roi_livegrid = LivePlot(y=roi_livegrid_key)

    # poke the struck settings
    yield from bps.mv(sclr.mcas.prescale, prescale)
    yield from bps.mv(sclr.mcas.nuse, num_pixels)


    xspress3.fluor.shape = (num_pixels, 4, 4096)
    xspress3.fluor.dims = ("num_pixels", "channels", "bin_count")
    #xspress3.cam.trigger_mode.set(3)
    yield from bps.mv(xspress3.external_trig, True)
    #yield from mv(xspress3.total_points, num_pixels)
    yield from bps.mv(xspress3.hdf5.num_capture, num_pixels)
    #yield from mv(xspress3.settings.num_images, num_pixels)
    yield from bps.mv(xspress3.total_points, num_pixels)
    yield from bps.mv(xspress3.cam.num_images, num_pixels)

    @bpp.reset_positions_decorator([mono.linear])
    @bpp.stage_decorator([sclr])
    # @bpp.subs_decorator({"all": [roi_livegrid]})
    #@bpp.monitor_during_decorator([xs.channel01.mcaroi01.total_rbv.value])
    #@bpp.baseline_decorator([mono, xy_stage])
    # TODO put is other meta data
    @bpp.run_decorator(
        md={
            "scan_title": scan_title,
            "operator": operator,
            "user_input": {
                "element": element,
                "edge": edge,
                "start": start,
                "stop": stop,
                "step_size": step_size,
            },
            "derived_input": {
                "l_start": l_start,
                "l_stop": l_stop,
                "l_step_size": l_step_size,
                "lmres": lmres,
                "actual_l_step_size": a_l_step_size,
                "fly_velocity": flyspeed,
                "num_pixels": num_pixels,
                "prescale": prescale,
                "x": xy_stage.x.position,
                "y": xy_stage.y.position,
                "z": xy_stage.z.position
            },
        }
    )
    def fly_body():
        yield from bps.trigger_and_read([E_centers], name="energy_bins")

        @bpp.stage_decorator([x for x in [xspress3] if x is not None])
        def fly_once(y):
            # for y in range(num_scans):
            # go to start of row

            yield from bps.checkpoint()
            yield from bps.mv(mono.linear.velocity, 0.15)
            yield from bps.mv(mono.linear, l_start)
            yield from bps.sleep(0.2)
            # set the fly speed
            yield from bps.mv(mono.linear.velocity, flyspeed)
            yield from bps.sleep(0.2)
            yield from bps.trigger_and_read([mono], name="row_ends")
            yield from bps.sleep(0.2)
            for v in ["p1600=0", "p1600=1"]:
                yield from bps.mv(dtt, v)
                yield from bps.sleep(0.1)

            # arm the Struck
            yield from bps.trigger(sclr, group=f"fly_energy_{y}")
            yield from bps.trigger(xspress3, group=f"fly_energy_{y}")

            # TODO: Figure out why this sleep is needed.
            yield from bps.sleep(2)

            # fly the motor
            yield from bps.abs_set(
                mono.linear, l_stop + a_l_step_size, group=f"fly_energy_{y}"
            )
            yield from bps.wait(group=f"fly_energy_{y}")

            yield from bps.trigger_and_read([mono], name="row_ends")

            yield from bps.mv(mono.linear.velocity, flyspeed)
            # hard coded to let the sclr count its fingers and toes
            yield from bps.sleep(0.5)
            # read and save the struck
            yield from bps.create(name="primary")
            yield from bps.read(sclr)
            yield from bps.read(xspress3)
            yield from bps.save()
            yield from bps.mv(mono.linear.velocity, 0.15)
            yield from bps.sleep(0.2)

        for scan_iter in range(num_scans):
            #yield from bps.mv(mono.linear.velocity, 0.3)
            print(num_scans)
            yield from bps.mv(xspress3.fly_next, True)
            yield from fly_once(scan_iter)

    yield from fly_body()

#export data
    print("Waiting for files... ...")
    yield from bps.sleep(5)
    #export_E_fly_smart(-1)

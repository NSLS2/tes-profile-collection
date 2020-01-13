import pprint

from bluesky.plans import list_scan
from bluesky.plan_stubs import abs_set

from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np


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


def E_Step_Scan(scan_title, *, operator, element, dwell_time=3, E_sections, step_size, num_scans, xspress3=None):
#def E_Step_Scan(dwell_time,*, E_sections = [2700, 2800, 2900, 3200], step_size = [4, 1, 2], num_scans):

    E_sections = np.array(E_sections)
    step_size = np.array(step_size)

    ept = []
    for ii in range(step_size.shape[0]):
        ept = ept[0:-1]
        ept = np.append(ept, np.linspace(E_sections[ii], E_sections[ii+1], np.int((E_sections[ii+1] - E_sections[ii])/step_size[ii])+1))

    yield from abs_set(sclr.cnts.preset_time, dwell_time)
    yield from abs_set(xs.settings.acquire_time, dwell_time)

    suitcase_rr = RunRouter(e_step_serializer_factory)

    @bpp.reset_positions_decorator([mono.linear])
    @bpp.stage_decorator([sclr])
    @bpp.subs_decorator({"all": [suitcase_rr]})
    @bpp.monitor_during_decorator([xs.channel1.rois.roi01.value])
    @bpp.baseline_decorator([mono, xy_stage])
    # TODO put is other meta data
    @bpp.run_decorator(
        md={
            "scan_title": scan_title,
            "operator": operator,
            "user_input": {
                "element": element,
                "E_sections": E_sections,
                "dwell_time": dwell_time,
                "step_size": step_size,
            },
            "derived_input": {

            },
        }
    )
    def scan_once():
        yield from list_scan([sclr,xs],mono.energy,ept)

    for scan_iter in range(num_scans):
        yield from scan_once()

    print("Waiting for files... ...")
    yield from bps.sleep(15)
    for scan_n in range(1, num_scans+1):
        artifacts = e_step_export(db[-1*scan_n])
        pprint.pprint(artifacts)


# np.fromstring('1, 2', dtype=int, sep=',')

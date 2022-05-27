import numpy as np
from bluesky.callbacks.broker import post_run
from bluesky.callbacks.mpl_plotting import LiveGrid
from bluesky.plans import outer_product_scan

# xrfmapTiffOutputDir = '/home/xf08bm/DATA2017/Comissioning/20170619/'
# hard-coded for testing now; need to be set to automatically use SAF, today's date, etc.


def xrfmap(
    *,
    xstart,
    xnumstep,
    xstepsize,
    ystart,
    ynumstep,
    ystepsize,
    rois=(),
    # shutter=True,
    # align=False,
    # acqtime,
    # numrois=1,
    # i0map_show=True,
    # itmap_show=False,
    # record_cryo=False,
    # setenergy=None,
    # u_detune=None,
    # echange_waittime=10
):
    """
    input:
         xstart, xnumstep, xstepsize (float)
         ystart, ynumstep, ystepsize (float)

    """
    # define detector used for xrf mapping functions
    xrfdet = [sclr]  # currently only the scalar; to-do: save full spectra

    xstop = xstart + xnumstep * xstepsize
    ystop = ystart + ynumstep * ystepsize

    # setup live callbacks:

    livetableitem = [xy_stage.x, xy_stage.y]
    livecallbacks = []

    for roi in rois:
        livecallbacks.append(
            LiveGrid(
                (ynumstep + 1, xnumstep + 1),
                roi,
                xlabel="x (mm)",
                ylabel="y (mm)",
                extent=[xstart, xstop, ystart, ystop],
            )
        )
        livetableitem.append(roi)

    #     # setup LiveOutput
    #     xrfmapOutputTiffTemplate = (xrfmapTiffOutputDir +
    #                                 "xrfmap_scan{start[scan_id]}" +
    #                                 roi + ".tiff")
    #     # xrfmapTiffexporter = LiveTiffExporter(roi, xrfmapOutputTiffTemplate, db=db)
    #     xrfmapTiffexporter = RasterMaker(xrfmapOutputTiffTemplate, roi)
    #     livecallbacks.append(xrfmapTiffexporter)

    livecallbacks.append(LiveTable(livetableitem))

    # setup LiveOutput

    # if sclr in xrfdet:
    #     for sclrDataKey in [getattr(sclr.cnts.channels, f'chan{j:02d}') for d in range(1, 21)]:
    #         xrfmapOutputTiffTemplate = (xrfmapTiffOutputDir +
    #                                     "xrfmap_scan{start[scan_id]}" +
    #                                     sclrDataKey + ".tiff")
    #
    #         # xrfmapTiffexporter = LiveTiffExporter(roi, xrfmapOutputTiffTemplate, db=db)
    #
    #         # LiveTiffExporter exports one array from one event,
    #         # commented out for future reference
    #         xrfmapTiffexporter = RasterMaker(xrfmapOutputTiffTemplate,
    #                                          sclrDataKey)
    #         livecallbacks.append(xrfmapTiffexporter)

    xrfmap_scanplan = outer_product_scan(
        xrfdet,
        xy_stage.y,
        ystart,
        ystop,
        ynumstep + 1,
        xy_stage.x,
        xstart,
        xstop,
        xnumstep + 1,
        False,
    )
    xrfmap_scanplan = bp.subs_wrapper(xrfmap_scanplan, livecallbacks)

    scaninfo = yield from xrfmap_scanplan

    return scaninfo


def test():
    while I0 < 0.1:
        print ("Low current")
        yield from sleep (1)
    while I0 > 0.1:
        print("Good current")


# Functions for custom scans using KB-mirrors and vstream cam:

@bpp.stage_decorator([vstream, I0, ring_current, kbh.dsh])
@bpp.run_decorator()
def myplan():
    start_pos = 3.34
    yield from bps.mv(kbh.dsh, start_pos-0.10)
    yield from bps.abs_set(kbh.dsh, start_pos+0.15, wait=False, group="mover")
    yield from bps.trigger_and_read([vstream, I0, ring_current, kbh.dsh])
    yield from bps.wait(group="mover")
    yield from bps.mv(kbh.dsh, start_pos)


def get_random_walk(n, n_cycles, range=[-1, 1]):

    res = np.real(np.fft.ifft(np.fft.fft(np.random.standard_normal(size=n)) * np.exp(-np.abs(np.fft.fftfreq(n, n_cycles/n)))))
    res -= res.mean()

    res_quartiles = np.percentile(res,q=[5,95])
    res *= np.diff(range) / np.diff(res_quartiles)
    res += range[0] - np.percentile(res,q=5)

    return res

# motor_ranges = 4*[[-.1,.1]]

# plt.figure()
# motor_positions = []
# for motor_range in motor_ranges:
#     motor_positions.append(get_random_walk(n=1000,n_cycles=10,range=motor_range))

#     plt.plot(motor_positions[-1])


def kb_trajectories(n, n_cycles, motors_ranges={}):
    trajectories = {}
    for mirror, motor_dict in motors_ranges.items():
        for motor_name, rng in motor_dict.items():
            trajectories[getattr(mirror, motor_name).name] = get_random_walk(n, n_cycles, range=rng)
    return trajectories

# Example run:
# kb_traj_list = kb_trajectories(1000, 16, ranges=[kbh_ranges["dsh"], kbh_ranges["ush"], kbv_ranges["dsh"], kbv_ranges["ush"]])


def scan_with_random_walk(detectors=[vstream, I0, ring_current],
                          motors_ranges={kbh: {"dsh": [-0.1/2, 0.15/2], "ush": [-0.2/2, 0.15/2]},
                                         kbv: {"dsh": [-0.2/2, 0.15/2], "ush": [-0.2/2, 0.25/2]}},
                          num_points=50000,
                          num_cycles=800,
                          testing=False,
                          ):
    """
    kbh_ranges = {"dsh": [-0.1/2, 0.15/2], "ush": [-0.2/2, 0.15/2]}
    kbv_ranges = {"dsh": [-0.2/2, 0.15/2], "ush": [-0.2/2, 0.25/2]}

    kb_traj_list = kb_trajectories(50000, 800, ranges=[kbh_ranges["dsh"], kbh_ranges["ush"], kbv_ranges["dsh"], kbv_ranges["ush"]])

    md = {"plan_args": {"detectors": [vstream.name, I0.name, ring_current.name],
                        "args": [kbh.dsh.name, kbh.ush.name, kbv.dsh.name, kbv.ush.name,
                                 'kb_trajectories(50000, 800, ranges=[kbh_ranges["dsh"], kbh_ranges["ush"], kbv_ranges["dsh"], kbv_ranges["ush"]])']},
          "plan_pattern_args": "See 'plan_args'"}

    RE(bp.rel_list_scan([vstream, I0, ring_current], kbh.dsh, kb_traj_list[0], kbh.ush, kb_traj_list[1], kbv.dsh, kb_traj_list[2], kbv.ush, kb_traj_list[3], md=md))
    """
    traj_dict = kb_trajectories(num_points, num_cycles, motors_ranges=motors_ranges)

    detectors_str = [detector.name for detector in detectors]

    motors = [getattr(mirror, motor_name) for mirror, motor_dict in motors_ranges.items()
                                          for motor_name in motor_dict]
    motors_str = [motor.name for motor in motors]

    args = []
    for motor in motors:
        if testing:
            args += [motor.name, traj_dict[motor.name]]  # testing
        else:
            args += [motor, traj_dict[motor.name]]  # production

    motors_ranges_str = {mirror.name: motor_dict for mirror, motor_dict in motors_ranges.items()}

    md = {"plan_name": "scan_with_random_walk",
          "plan_args": {"detectors": [det.name for det in detectors],
                        "args": [motors_str,
                                 f'kb_trajectories({num_points}, {num_cycles}, motors_ranges={motors_ranges_str}']},
          "plan_pattern_args": "See 'plan_args'"}

    if testing:
        print(f"bp.rel_list_scan({detectors_str}, *{args}, md={md})")  # testing
    else:
        yield from bp.rel_list_scan(detectors, *args, md=md)  # production


# Example:
"""
RE(scan_with_random_walk(detectors=[vstream, I0, ring_current],
                         motors_ranges={kbh: {"dsh": [-0.1/2, 0.15/2], "ush": [-0.2/2, 0.15/2]},
                                        kbv: {"dsh": [-0.2/2, 0.15/2], "ush": [-0.2/2, 0.25/2]}},
                         num_points=50000,
                         num_cycles=800))
"""

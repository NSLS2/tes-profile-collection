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

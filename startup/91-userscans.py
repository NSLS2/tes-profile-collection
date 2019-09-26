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

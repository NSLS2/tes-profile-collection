import matplotlib.pyplot as plt
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.callbacks import LiveTable, LivePlot


from bluesky.plans import list_scan
import bluesky.plans as bp
from bluesky.plan_stubs import mv
from bluesky.plan_stubs import one_1d_step
from bluesky.preprocessors import finalize_wrapper
from bluesky.preprocessors import subs_wrapper
from bluesky.utils import short_uid as _short_uid
#import scanoutput
import numpy
import time
from epics import PV
from databroker import get_table
import collections



def escan():
    """
    Scan the mono_energy while reading the scaler.

    Parameters
    ----------
    start : number
    stop : number
    num : integer
        number of data points (i.e. number of strides + 1)
    md : dictionary, optional
    """

    """
    dets = [xs]
    motor = mono.energy
    cols = ['I0', 'fbratio', 'It', 'If_tot']
    x = 'mono_energy'
    fig, axes = plt.subplots(2, sharex=True)
    plan = bp.scan(dets, motor, start, stop, num, md=md)
    plan2 = bpp.subs_wrapper(plan, [LiveTable(cols),
                                    LivePlot('If_tot', x, ax=axes[0]),
                                    LivePlot('I0', x, ax=axes[1])])
    yield from plan2
    """
    ept = numpy.array([])
    det = [sclr,xs]

    last_time_pt = time.time()
    ringbuf = collections.deque(maxlen=10)
    #c2pitch_kill=EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")
    xs.external_trig.put(False)

    #@bpp.stage_decorator([xs])
    yield from abs_set(xs.settings.acquire_time,0.1)
    yield from abs_set(xs.total_points,100)


    roi_name = 'roi{:02}'.format(roinum[0])
    roi_key = []
    roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
    livetableitem.append(roi_key[0])
    livecallbacks.append(LiveTable(livetableitem))
    liveploty = roi_key[0]
    liveplotx = energy.energy.name
    liveplotfig = plt.figure('raw xanes')
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))

    myscan = count
import matplotlib.pyplot as plt
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.callbacks import LiveTable, LivePlot


def escan(start, stop, num, md=None):
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
    dets = [sclr]
    motor = mono.energy
    cols = ['I0', 'fbratio', 'It', 'If_tot']
    x = 'mono_energy'
    fig, axes = plt.subplots(2, sharex=True)

    plan = bp.scan(dets, motor, start, stop, num, md=md)
    plan2 = bpp.subs_wrapper(plan, [LiveTable(cols),
                                   LivePlot('If_tot', x, ax=axes[0]),
                                   LivePlot('I0', x, ax=axes[1])])
    yield from plan2

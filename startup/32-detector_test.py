print(f"Loading {__file__!r} ...")

import os
from bluesky.plans import count
from bluesky.plan_stubs import abs_set

from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import numpy as np
import pandas as pd



def detector_test(operator, scan_title, start_dwell_time, num_scans):
    yield from bps.mv(xs.external_trig, False)
    sclr.set_mode("counting")
    operator = operator
    scan_title = scan_title
    num_scans = num_scans
    start_dwell_time = start_dwell_time
    def scan_once(dwell_time):
        yield from bps.mv(sclr.cnts.preset_time, dwell_time,
                          xs.settings.acquire_time, dwell_time)
        yield from count([sclr, xs],1000)

    for scan_iter in range(num_scans):
        dwell_time = start_dwell_time * (scan_iter+1)/1000
        yield from scan_once(dwell_time)
        h = db[-1]
        I0 = h.table()['I0']
        # If = h.table()['xs_channel1_rois_roi01_value_sum']
        If = h.table()['xs3_channel01_mcaroi01_total_rbv']
        df = pd.DataFrame({'#I0': I0, 'If': If})
        df.to_csv('/home/xf08bm/Users/TEMP/test/' + f'{operator}-{scan_title}-{scan_iter}.csv')

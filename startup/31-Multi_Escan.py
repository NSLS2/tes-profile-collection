"""""

import pandas as pd
import numpy as np

def multi_Efly():

    path = "/home/xf08bm/Desktop/Users/Multi_Escan_setup.xls"
    data = np.array(pd.read_excel(path, index_col=0))

    for ii in range(data.shape[0]):
        x = data[ii, 0]
        y = data[ii, 1]
        z = data[ii, 2]
        scan_title = data[ii, 3]
        operator = data[ii, 4]
        element = data[ii, 5]
        start = data[ii, 6]
        stop = data[ii, 7]
        step_size = data[ii, 8]
        num_scans = data[ii, 9]
        yield from E_fly(scan_title, operator, element, start, stop, step_size, num_scans, xspress3=None)
    #yield from multi_Efly()
"""

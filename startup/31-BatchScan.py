import os
import pandas as pd
import numpy as np


def Batch_E_fly(index=None):
    # root = "/home/xf08bm/Desktop/Users/"
    # root.withdraw()

    # file_path = filedialog.askopenfilename()
    file_path = os.path.join(get_ipython().profile_dir.location, 'config/BatchScan_Para.xls')
    data = np.array(pd.read_excel(file_path, sheet_name="E_fly", index_col=0))
    xy_fly_stage = xy_stage

    if index is None:
        index = range(data.shape[0])
    for ii in index:
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
        flyspeed = data[ii, 11]
        yield from bps.mv(xy_fly_stage.x, x, xy_fly_stage.y, y, xy_fly_stage.z, z)
        yield from E_fly(
            scan_title,
            operator=operator,
            element=element,
            start=start,
            stop=stop,
            step_size=step_size,
            num_scans=num_scans,
            flyspeed=flyspeed,
            xspress3=xs,
        )


def Batch_xy_fly(index=None):
    # root = "/home/xf08bm/Desktop/Users/"
    # root.withdraw()

    # file_path = filedialog.askopenfilename()
    file_path = os.path.join(get_ipython().profile_dir.location, 'config/BatchScan_Para.xls')
    data = np.array(pd.read_excel(file_path, sheet_name="xy_fly", index_col=0).dropna())
    xy_fly_stage = xy_stage
    e_back = yield from _get_v_with_dflt(mono.e_back, 1977.04)
    energy_cal = yield from _get_v_with_dflt(mono.cal, 0.40118)
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
    if index is None:
        index = range(data.shape[0])
    # @bpp.reset_positions_decorator([mono.linear])
    for ii in index:
        x = data[ii, 0]
        y = data[ii, 1]
        z = data[ii, 2]
        scan_title = data[ii, 9]
        operator = data[ii, 10]
        xstart = data[ii, 3]
        xstop = data[ii, 4]
        xstep_size = data[ii, 5]
        ystart = data[ii, 6]
        ystop = data[ii, 7]
        ystep_size = data[ii, 8]
        dwell_time = data[ii, 11]
        E_e = data[ii, 12]
        detector = data[ii, 13]
        yield from bps.mv(xy_fly_stage.x, x, xy_fly_stage.y, y, xy_fly_stage.z, z)
        yield from bps.sleep(2)
        l_start = _energy_to_linear([E_e])
        yield from bps.mv(mono.linear, l_start)
        yield from xy_fly(
            scan_title=scan_title,
            beamline_operator=operator,
            dwell_time=dwell_time,
            xstart=xstart,
            xstop=xstop,
            xstep_size=xstep_size,
            ystart=ystart,
            ystop=ystop,
            ystep_size=ystep_size,
            xspress3=xs,
        )

def Batch_E_step(index=None):
    # root = "/home/xf08bm/Desktop/Users/"
    # root.withdraw()

    # file_path = filedialog.askopenfilename()
    file_path = os.path.join(get_ipython().profile_dir.location, 'config/BatchScan_Para.xls')
    data = np.array(pd.read_excel(file_path, sheet_name="E_step", index_col=0).dropna())
    xy_fly_stage = xy_stage

    if index is None:
        index = range(data.shape[0])
    for ii in index:
        x = data[ii, 0]
        y = data[ii, 1]
        z = data[ii, 2]
        scan_title = data[ii, 3]
        operator = data[ii, 4]
        element = data[ii, 5]
        dwell_time = data[ii, 10]
        E_selections = list(map(float,data[ii, 8].strip('][').split(',')))
        Step_size = list(map(float,data[ii, 9].strip('][').split(',')))
        num_scans = data[ii, 6]

        yield from bps.mv(xy_fly_stage.x, x, xy_fly_stage.y, y, xy_fly_stage.z, z)
        yield from E_Step_Scan(scan_title=scan_title,
                               operator=operator,
                               element=element,
                               dwell_time=dwell_time,
                               E_sections=E_selections,
                               step_size=Step_size,
                               num_scans=num_scans,
                               xspress3=xs)
def Batch_XAS_mapping(index=None):

    file_path = os.path.join(get_ipython().profile_dir.location, 'config/BatchScan_Para.xls')
    data = np.array(pd.read_excel(file_path, sheet_name="E_step", index_col=0))
    if index is None:
        index = range(data.shape[0])
    for ii in index:
        x = data[ii, 0]
        y = data[ii, 1]
        z = data[ii, 2]
        scan_title = data[ii, 3]
        operator = data[ii, 4]
        element = data[ii, 5]
        dwell_time = data[ii, 10]
        E_sections = list(map(float,data[ii, 8].strip('][').split(',')))
        Step_size = list(map(float,data[ii, 9].strip('][').split(',')))
        num_scans = data[ii, 6]

        yield from bps.mv(xy_fly_stage.x, x, xy_fly_stage.y, y, xy_fly_stage.z, z)

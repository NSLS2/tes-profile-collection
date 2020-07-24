def Batch_E_fly(index=None):
    # root = "/home/xf08bm/Desktop/Users/"
    # root.withdraw()

    # file_path = filedialog.askopenfilename()
    file_path = "/home/xf08bm/Desktop/Users/BatchScan_Para.xls"
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
    file_path = "/home/xf08bm/Desktop/Users/BatchScan_Para.xls"
    data = np.array(pd.read_excel(file_path, sheet_name="xy_fly", index_col=0))
    xy_fly_stage = xy_stage

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
        yield from bps.mv(mono.energy, E_e)
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

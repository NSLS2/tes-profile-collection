import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit

def nano_knife_edge(motor, start, stop, stepsize, acqtime,
                    normalize=True, use_trans=False,
                    scan_only=False, shutter=True, plot=True, plot_guess=False):
    """
    motor       motor   motor used for scan
    start       float   starting position
    stop        float   stopping position
    stepsize    float   distance between data points
    acqtime     float   counting time per step
    fly         bool    if the motor can fly, then fly that motor
    high2low    bool    scan from high transmission to low transmission
                        ex. start will full beam and then block with object (knife/wire)
    """

    # Need to convert stepsize to number of points
    num = np.round((stop - start) / stepsize) + 1

    # Run the scan
    if (motor.name == 'nano_stage_sx'):
        fly = True
        pos = 'enc1'
        fluor_key = 'fluor'
        y0 = nano_stage.sy.user_readback.get()
        plotme = LivePlot('')

        @subs_decorator(plotme)
        def _plan():
            yield from nano_scan_and_fly(start, stop, num,
                                         y0, y0, 1, acqtime,
                                         shutter=shutter)

        yield from _plan()
    elif (motor.name == 'nano_stage_sy'):
        fly = True
        pos = 'enc2'
        fluor_key = 'fluor'
        x0 = nano_stage.sx.user_readback.get()
        plotme = LivePlot('')

        @subs_decorator(plotme)
        def _plan():
            yield from nano_y_scan_and_fly(start, stop, num,
                                           x0, x0, 1, acqtime,
                                           shutter=shutter)

        yield from _plan()
    elif (motor.name == 'xy_stage_x'):
        fly = False
        pos = motor.name
        fluor_key = 'xs_channel1'
        y0 = xy_stage.y.user_readback.get()
        dets = [xs, sclr]
        yield from abs_set(xs.total_points, num)
        livecallbacks = [
            LiveTable(
                [
                    motor.name,
                    xs.channel1.rois.roi01.value.name
                ]
            )
        ]

        livecallbacks.append(
            LivePlot(
                xs.channel1.rois.roi01.value.name,
                motor.name
            )
        )


        yield from subs_wrapper(scan(dets, motor, start, stop, num),
                                {'all': livecallbacks})

    elif (motor.name == 'xy_stage_y'):
        fly = False
        pos = motor.name
        fluor_key = 'xs_channel1'
        x0 = xy_stage.x.user_readback.get()
        dets = [xs, sclr]
        yield from abs_set(xs.total_points, num)
        livecallbacks = [LiveTable([motor.name,
                                    xs.channel1.rois.roi01.value.name])]
        livecallbacks.append(LivePlot(xs.channel1.rois.roi01.value.name,
                                      motor.name))

        yield from subs_wrapper(scan(dets, motor, start, stop, num),
                                {'all': livecallbacks})

    else:
        print(f'{motor.name} is not implemented in this scan.')
        return

    #plot_knife_edge(scanid=db[-1].start['scan_id'], plot_guess=False)



def plot_knife_edge(scanid=-1, fluor_key='fluor', use_trans=False, normalize=True, plot_guess=True,
                    bin_low=None, bin_high=None, plotme=None):
    def f_offset_erf(x, A, sigma, x0, y0):
        x_star = (x - x0) / sigma
        return A * erf(x_star / np.sqrt(2)) + y0

    def f_two_erfs(x, A1, sigma1, x1, y1, A2, sigma2, x2, y2):
        x1_star = (x - x1) / sigma1
        x2_star = (x - x2) / sigma2

        f_combo = f_offset_erf(x, A1, sigma1, x1, y1) + f_offset_erf(x, A2, sigma2, x2, y2)
        return f_combo

    # Get the scanid

    h = db[int(scanid)]
    id_str = h.start['scan_id']
    fly = False

    # Get the information from the previous scan
    haz_data = False
    loop_counter = 0
    MAX_LOOP_COUNTER = 1
    print('Waiting for data...', end='', flush=True)
    while (loop_counter < MAX_LOOP_COUNTER):
        try:
            if (fly):
                tbl = db[int(id_str)].table('stream0', fill=True)
            else:
                tbl = h.table()
            haz_data = True
            #print(tbl)
            break
        except:
            loop_counter += 1
            ttime.sleep(1)

    # Check if we haz data
    if (not haz_data):
        print('Data collection timed out!')
        return

    # Get the data

    I0 = h.table()['I0']
    If_1_roi1 = h.table()['xs_channel1_rois_roi01_value_sum']
    y = If_1_roi1 / I0 * 1000000
   # y = I0/If_1_roi1
    try:
        x = np.array(h.table()['xy_stage_x'])
    except:
        x = np.array(h.table()['xy_stage_y'])
    y = np.array(y)

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dydx = np.gradient(y, x)
    # try:
    #    hf = h5py.File('/home/xf05id1/current_user_data/knife_edge_scan.h5', 'a')
    #    tmp_str = 'dataset_%s' % id_str
    #    hf.create_dataset(tmp_str, data=[d,y,x,y]) #raw cts, norm_cts, x_pos, y_pos
    #    hf.close()
    #    ftxt = open('/home/xf05id1/current_user_data/knife_edge_scan.txt','a')
    #    ftxt.write(data=[d,y,x,y])
    #    ftxt.close()
    # except:
    #    pass

    # Fit the raw data
    # def f_int_gauss(x, A, sigma, x0, y0, m)
    # def f_offset_erf(x, A, sigma, x0, y0):
    # def f_two_erfs(x, A1, sigma1, x1, y1,
    #                   A2, sigma2, x2, y2):
    p_guess = [0.5 * np.amax(y),
               0.500,
               x[np.argmax(y)] - 1.0,
               np.amin(y),
               -0.5 * np.amax(y),
               0.500,
               x[np.argmax(y)] + 1.0,
               np.amin(y)]
    try:
        # popt, _ = curve_fit(f_offset_erf, x, y, p0=p_guess)
        popt, _ = curve_fit(f_two_erfs, x, y, p0=p_guess)
    except:
        print('Raw fit failed.')
        popt = p_guess

    C = 2 * np.sqrt(2 * np.log(2))
    cent_position = (popt[2] + popt[6]) / 2
    print(f'The beam size is {C * popt[1]:.4f} mm')
    print(f'The beam size is {C * popt[5]:.4f} mm')

    # print(f'\nThe left edge is at\t{popt[2]:.4f}.')
    # print(f'The right edge is at\t{popt[6]:.4f}.')
    print(f'The center is at\t{(popt[2] + popt[6]) / 2:.4f}.')

    # Plot variables
    x_plot = np.linspace(np.amin(x), np.amax(x), num=100)
    y_plot = f_two_erfs(x_plot, *popt)
    # y_plot = f_offset_erf(x_plot, *popt)
    dydx_plot = np.gradient(y_plot, x_plot)

    # Display fit of raw data
    if (plotme is None):
        fig, ax = plt.subplots()
    else:
        ax = plotme.ax

    ax.cla()
    ax.plot(x, y, '*', label='Raw Data')
    if (plot_guess):
        ax.plot(x_plot, f_two_erfs(x_plot, *p_guess), '--', label='Guess fit')
    ax.plot(x_plot, y_plot, '-', label='Final fit')
    ax.set_title(f'Scan {id_str}')
    ax.set_xlabel('x or y')
    if (normalize):
        ax.set_ylabel('Normalized ROI Counts')
    else:
        ax.set_ylabel('ROI Counts')
    ax.legend()

    return cent_position
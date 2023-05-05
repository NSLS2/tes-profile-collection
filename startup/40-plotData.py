import numpy as np
import  matplotlib.pyplot as plt


def pltxas(scanID = -1, mode = 'TEY'):
    scanID = scanID
    mode = mode
    h = db[scanID]
    start = h.start
    element = start['user_input']['element']
    plan_name = start['plan_name'],
    roi = rois(element)
    I_TEY = h.table()['fbratio']
    I0 = h.table()['I0']
    print(plan_name)
    if plan_name == 'Batch_E_fly':
        d = np.array(list(h.data('fluor', stream_name='primary', fill=True)))
        If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)
        E = h.table('energy_bins')['E_centers'][1]

    else:
        E = h.table()['mono_energy']
        I0 = h.table()['I0']
        I_TEY = h.table()['fbratio']
        # If = h.table()['xs_channel1_rois_roi01_value_sum']
        If = h.table()['xs3_channel01_mcaroi01_total_rbv']
    print(E)
    print(If)
    if mode == 'fluo':
        plt.plot(E,If/I0)
    elif mode == 'TEY':
        plt.plot(E,I_TEY/I0)
    else:
        print('mode not supported yet')





'''''''''
def pltstageX(scanID = -1, mode = 'fluo'):
    scanID = scanID
    mode = mode
    h = db[scanID]
    start = h.start
    element = start['user_input']['element']
    plan_name = start['plan_name']
    roi = rois(element)
    I0 = h.table()['IO']

    print(plan_name == 'X_step_scan')
    if plan_name == 'X_step_scan':
        d = np.array(list(h.data('fluor', stream_name='primary', fill=True)))
        If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)
        x = h.table('xy_stage_x')

    else:
        x = h.table()['xy_stage_x']
        I0 = h.table()['I0']
        If = h.table()['xs_channel1_rois_roi01_value_sum']
    print(E)
    print(If)
    if mode == 'fluo':
        plt.plot(x, If / I0)

    else:
        print('mode not supported yet')
        
 '''''''''

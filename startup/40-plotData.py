print(f"Loading {__file__!r} ...")

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
    I_TEY = h['primary']['data']['fbratio'].read()
    I0 = h['primary']['data']['I0'].read()
    print(plan_name)
    if plan_name == 'Batch_E_fly':
        d = h['primary']['data']['fluor'].read()
        If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)
        E = h.table('energy_bins')['E_centers'][1]

    else:
        E = h['baseline']['data']['mono_energy'].read()
        I0 = h['primary']['data']['I0'].read()
        I_TEY = h['primary']['data']['fbratio'].read()
        # If = h['primary']['data']['xs_channel1_rois_roi01_value_sum']
        # TODO: This doesn't work yet. Need to update for tiled.
        If = h['primary']['data']['xs3_channel01_mcaroi01_total_rbv'].read()
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
    I0 = h['primary']['data']['IO'].read()

    print(plan_name == 'X_step_scan')
    if plan_name == 'X_step_scan':
        d = h['primary']['data']['fluor'].read()
        If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)
        x = h['xy_stage_x'].read()

    else:
        x = h['primary']['data']['xy_stage_x'].read()
        I0 = h['primary']['data']['I0'].read()
        If = h['primary']['data']['xs_channel1_rois_roi01_value_sum'].read()
    print(E)
    print(If)
    if mode == 'fluo':
        plt.plot(x, If / I0)

    else:
        print('mode not supported yet')
        
 '''''''''

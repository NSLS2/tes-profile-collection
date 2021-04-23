import numpy as np
import pandas as pd
import datetime

def export_E_fly(scanID=-1):
    h = db[scanID]
    start = h.start
    element = start['user_input']['element']
    roi = rois(element)
    d = np.array(list(h.data('fluor', stream_name='primary', fill=True)))
    If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)
    E = h.table('energy_bins')['E_centers'][1]
    I0 = h.table()['I0']
    Dwell_time = h.table()['dwell_time']
    dt = datetime.datetime.fromtimestamp(start["time"])

    file_head = {'beamline_id': 'TES/8-BM of NSLS-II',
                 'operator': start['operator'],
                 'plan_name': start['plan_name'],
                 'scan_id': start['scan_id'],
                 'scan_title': start['scan_title'],
                 'time': f"{dt.date().isoformat()} {dt.time().isoformat()}",
                 'uid': start['uid'],
                 'user_input': start['user_input'],
                 'derived_input': start['derived_input']
                 }

    for ii in range(If.shape[0]):
        if If.shape[2] == 1:
            df = pd.DataFrame({'#Energy': E,
                               'Dwell_time': Dwell_time[ii + 1],
                               'I0': I0[ii + 1],
                               'If_CH1': If[ii, :, 0],
                               })
        else:
            df = pd.DataFrame({'#Energy': E,
                               'Dwell_time': Dwell_time[ii + 1],
                               'I0': I0[ii + 1],
                               'If_CH1': If[ii, :, 0],
                               'If_CH2': If[ii, :, 1]
                               })

        filepath = os.path.expanduser(
            f"~/Users/Data/{start['operator']}/{dt.date().isoformat()}/E_fly/"
            f"{start['scan_title']}-{start['scan_id']}-{start['operator']}-{dt.time().strftime('%H-%M-%S')}-{ii}.dat")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wt") as output_file:
            output_file.write(pprint.pformat(file_head))
            output_file.write('\n')
            output_file.write('\n')
            output_file.write('\n')

        df.to_csv(filepath, header=True, index=False, mode='a')
        print(f'Data exported to {filepath}')


def export_E_step(scanID=-1, scan_iter=0):
    h = db[scanID]
    E = h.table()['mono_energy']
    I0 = h.table()['I0']
    If_1_roi1 = h.table()['xs_channel1_rois_roi01_value_sum']
    If_1_roi2 = h.table()['xs_channel1_rois_roi02_value_sum']
    If_1_roi3 = h.table()['xs_channel1_rois_roi03_value_sum']
    If_1_roi4 = h.table()['xs_channel1_rois_roi04_value_sum']
    If_2_roi1 = h.table()['xs_channel2_rois_roi01_value_sum']
    If_2_roi2 = h.table()['xs_channel2_rois_roi02_value_sum']
    If_2_roi3 = h.table()['xs_channel2_rois_roi03_value_sum']
    If_2_roi4 = h.table()['xs_channel2_rois_roi04_value_sum']

    df = pd.DataFrame({'#Energy': E, 'I0': I0,
                       'If_CH1_roi1': If_1_roi1, 'If_CH1_roi2': If_1_roi2, 'If_CH1_roi3':If_1_roi3, 'If_CH1_roi4': If_1_roi4,
                       'If_CH2_roi1': If_2_roi1, 'If_CH2_roi2': If_2_roi2, 'If_CH2_roi3':If_2_roi3, 'If_CH2_roi4': If_2_roi4})
    start = h.start
    dt = datetime.datetime.fromtimestamp(start["time"])

    file_head ={'beamline_id': 'TES/8-BM of NSLS-II',
     'operator': start['operator'],
     'plan_name': start['plan_name'],
     'scan_id': start['scan_id'],
     'scan_title': start['scan_title'],
     'time': f"{dt.date().isoformat()} {dt.time().isoformat()}",
     'uid': start['uid'],
     'user_input': start['user_input'],
     'derived_input': start['derived_input']
                }

    filepath = os.path.expanduser(
        f"~/Users/Data/{start['operator']}/{dt.date().isoformat()}/E_step/"
        f"{start['scan_title']}-{start['scan_id']}-{start['operator']}-{dt.time().strftime('%H-%M-%S')}-{scan_iter}.cvs")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wt") as output_file:
        output_file.write(pprint.pformat(file_head))
        output_file.write('\n')
        output_file.write('\n')
        output_file.write('\n')

    df.to_csv(filepath, header = True,index = False,mode='a')
    print(f'Data exported to {filepath}')

def tes_data(scanID = -1,scan_iter = 0):
    h = db[scanID]
    start = h.start
    if start['plan_name'] == 'E_fly' or start['plan_name'] == 'Batch_E_fly':
        export_E_fly(scanID)
    elif start['plan_name'] == 'list_scan':
        export_E_step(scanID,scan_iter)
    else:
        print(f"Plan_name is {start['plan_name']}.")

'''


def ResaveSclr(element, scan_title, scanID, operator):

    h = db[scanID]
    start = db[scanID].start
    If = h.table()[element]
    E = h.table('energy_bins')['E_centers'][1]

    I0 = h.table()['I0']
    Dwell_time = h.table()['dwell_time']

    dt = datetime.datetime.fromtimestamp(start["time"])

    for ii in range(If.shape[0]):
        df = pd.DataFrame({'#Energy': E,
                           'Dwell_time': Dwell_time[ii + 1],
                           'I0': I0[ii + 1],
                           'If': If[ii + 1]})
        filepath = os.path.expanduser(
            f"~/Users/Data/{start['operator']}/{dt.date().isoformat()}/xy_fly/"
            f"{start['scan_title']}-{start['scan_id']}-{start['operator']}-{dt.time().isoformat()}-{ii}.dat")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath)

'''

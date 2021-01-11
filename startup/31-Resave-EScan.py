import numpy as np
import pandas as pd


def Resave(element, file_name, scanID):
    h = db[scanID]
    roi = rois(element)

    if not roi:
        print("Wrong element, please check!")
        return
    else:
        d = np.array(list(h.data('fluor', stream_name='primary', fill=True)))

        If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)

        E = h.table('energy_bins')['E_centers'][1]

        I0 = h.table()['I0']
        Dwell_time = h.table()['dwell_time']

        for ii in range(If.shape[0]):
            df = pd.DataFrame({'#Energy': E,
                               'Dwell_time': Dwell_time[ii + 1],
                               'I0': I0[ii + 1],
                               'If': If[ii, :, 0]})
            df.to_csv('/home/xf08bm/Users/TEMP/' + f'{file_name}-{ii}.csv')

        print('Please go /home/xf08bm/Desktop/Users/TEMP/ to copy your data ASAP!')

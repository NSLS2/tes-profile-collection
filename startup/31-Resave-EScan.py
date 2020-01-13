import numpy as np
import pandas as pd


def Resave(element, file_name):
    h = db[-1]
    roi = rois(element)

    if not roi:
        print("Wrong element, please check!")
        return
    else:
        d = np.array(list(h.data('fluor', stream_name='primary', fill=True)))

        If = np.sum(d[:, :, :, roi[0]:roi[1]], axis=-1)

        E = h.table('energy_bins')['E_centers'][1]

        I0 = h.table()['I0']

        for ii in range(If.shape[0]):
            df = pd.DataFrame({'#Energy': E,
                               'I0': I0[ii + 1],
                               'If': If[ii, :, 0]})
            df.to_csv('/home/xf08bm/Users/TEMP/' + f'{file_name}-{ii}.csv')

        print('Please go /home/xf08bm/Desktop/Users/TEMP/ to copy your data ASAP!')


def rois(element):
    element = element.lower()
    if element == 's':
        roi = [221, 239]
    elif element == 'p':
        roi = [192, 210]
    elif element == 'ca':
        roi = []
    elif element == 'k':
        roi = []
    elif element == 'ar':
        roi = []
    elif element == 'Cl':
        roi = [253, 271]
    elif element == 'si':
        roi = []
    elif element == 'al':
        roi = []
    elif element == 'mg':
        roi = []
    elif element == 'pd':
        roi = [274, 292]
    elif element == 'au':
        roi = [202, 220]
    elif element == 'u':
        roi = []
    elif element == 'ru_croft':
        roi = [240, 280]
    elif element == 'y_croft':
        roi = [170, 205]
    elif element == 'zr_croft':
        roi = [190, 225]
    elif element == 'zr_sahiner':
        roi = [195, 225]
    else:
        roi = []

    return roi

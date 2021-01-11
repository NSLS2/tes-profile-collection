import matplotlib.pyplot as pp

element_to_roi = {
    "au": (202, 220),
    "ca": (364, 380),
    "s": (224, 240),
    "p": (192, 210),
    "al": (160,182),
    "mo": (240,258),
    "pd": (274, 292),
    "pt":(197,214),
    "cl": (253, 271),
    "ru_croft": (240, 280),
    "pd-2": (275, 295),
    "u": (310, 325),
    "ir":(190,206),
    "in":(324,340),
    "ti":(450,466),
    "y_croft": (170, 205),
    "zr_croft": (190, 225),
    "zr_sahiner": (195, 225),
}

element_to_enery = {
    "s": (2472),
}

def set_rois(element):
    roi = rois(element)


def rois(element):
    return element_to_roi[element.lower()]

    # if element == 's':
    #    roi = [221, 239]
    # elif element == 'p':
    #     roi = [192, 210]
    # elif element == 'ca':
    #     roi = []
    # elif element == 'k':
    #     roi = []
    # elif element == 'ar':
    #     roi = []
    # elif element == 'Cl':
    #     roi = [253, 271]
    # elif element == 'si':
    #     roi = []
    # elif element == 'al':
    #     roi = []
    # elif element == 'mg':
    #     roi = []
    # elif element == 'pd':
    #     roi = [274, 292]
    # elif element == 'au':
    #     roi = [202, 220]
    # elif element == 'u':
    #     roi = []
    # elif element == 'ru_croft':
    #     roi = [240, 280]
    # elif element == 'y_croft':
    #     roi = [170, 205]
    # elif element == 'zr_croft':
    #     roi = [190, 225]
    # elif element == 'zr_sahiner':
    #     roi = [195, 225]
    # else:
    #     roi = []
    #
    # return roi


#def plotScans():
    #    h = db[-1]
    #E = h.table()['mono_energy']
    #I0 = h.table()['I0']
    #If = h.table()['xs_channel1_rois_roi01_value_sum']
    #pp.plot(E, If)
    #pp.show()

print(f"Loading {__file__!r} ...")


# Element Format: element_edge, lower case only!!!
# Data in (): (roi low bin, rio width, absorption edge energy in eV)

element_to_roi = {
    "au": (202, 220),
    "ba":(1,1),
    "ca": (364, 380),
    "cd": (310,323),
    "cl": (256, 270),
    "k": (325, 345),
    "i": (388,403),
    "sn":(339,353),
    "p": (196, 208),
    "al": (160,182),
    "nb":(211,221),
    "mo": (224,238),
    "pb": (228,242),
    "pd": (274, 292),
    "pd_k": (274, 292),
    "pt":(197,214),
    "ru_l3": (256, 16, 2838),
    "rh_l3l2":(262, 290),
    "rh":(262, 278),
    "pd-2": (275, 295),
    "u": (310, 325),
    "ir":(190,206),
    "in":(324,340),
    "sc":(401,417),
    "ti":(447,461),
    "ti_kb":(490,502),
    "ta":(188,193),
    "y_croft": (170, 205),
    "zr": (190, 225),
    "zr_sahiner": (195, 225),
    "ag": (293, 307),
    "s_k": (224, 12),
    "sb": (356,372),
    "te":(372,386),
    "w":(200,218),
}

# Element Format: element_edge, lower case only!!!
# Data in (): (roi low bin, rio width, absorption edge energy in eV)
element_to_roi_smart = {
    "au_l3": (202, 220),
    "i_l3": (387, 13, 4557),
    "ba":(1,1),
    "ca_k": (362,14,4038),
    "cd": (310,323),
    "cl_k": (256,14,2822),
    "cs_l3": (421,16,5012),
    "k": (325, 345),
    "i": (388,403),
    "s": (224, 12,2472),
    "p_k": (196,10,2145.5),
    "nb_l3": (210,14,2371),
    "nb_l2": (219,14,2465),
    "nb_l1": (227,14,2698),
    "mo_l3": (224,14,2520),
    "mo_l2": (232,14,2625),
    "mo_l1": (240,14,2866),
    "pb_m3": (260,14,3066),
    "pb_m5": (228,2423),
    "pd_l3": (277,14,3173),
    "pd_l2": (292,14,3330),
    "pd_l1": (300,14,3604),
    "pt_m5":(197,17,2122),
    "ru_l3": (248,16,2838),
    "ru_l2": (260,16,2967),
    "ru_l1": (270,12,3224),
    "rh_l3": (263,14,3004),
    "rh_l2": (276,14,3146),
    "rh_l1": (285,14,3412),
    "u": (310, 325),
    "ir_m5":(190,14,2070),
    "in_l3":(324,14,3730),
    "pd_k":(278,12),
    "ti_k":(444, 12, 4966),
    "ti_kb":(490,502),
    "ta":(188,193),
    "y_croft": (170, 205),
    "zr": (190, 225),
    "zr_sahiner": (195, 225),
    "ag_l3": (291,14,3351),
    "ag_l2": (308,14,3524),
    "ag_l1": (316,14,3806),
    "s_k": (224, 12, 2472),
    "sc_k": (401, 16, 4492),
    "sn_l3": (339, 14, 3929),
    "sn_l2": (361, 14, 4156),
    "sb": (356,372),
    "te":(372,386),
    "w":(200,218),
}
# Format: element_edge, lower case only!!!


#def set_rois(element):
#    roi = rois(element)


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
    
    # return roi


#def plotScans():
    #    h = db[-1]
    #E = h.table()['mono_energy']
    #I0 = h.table()['I0']
    #If = h.table()['xs_channel1_rois_roi01_value_sum']
    #pp.plot(E, If)
    #pp.show()

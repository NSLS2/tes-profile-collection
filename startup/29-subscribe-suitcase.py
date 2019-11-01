import datetime
import os.path

import numpy as np


element_to_roi = {"s": (222, 240), "p": (192, 210), "pd": (274, 292)}

suitcase_config = """\
[versions]
"XDI"                         = "# XDI/1.0 Bluesky"

[columns]
"Column.1"                    = {column_label="energy",  data_key="E_centers", column_data="{[data][E_centers][0]}", units="eV"}
"Column.2"                    = {column_label="I0",      data_key="I0", column_data="{data[I0][0]}"}
"Column.3"                    = {column_label="If",      data_key="fluor", column_data="{data[fluor][0]}", transform="e_fly_roi1"}

[required_headers]
"Element.symbol"              = {data="{user_input[element]}", doc_name="start"}

[optional_headers]

"""


def e_fly_roi1(event_doc, roi_lo_ndx, roi_hi_ndx):
    # doc[data][fluor][0].shape is (243, 1, 4096)
    roi_sum = np.sum(event_doc["data"]["fluor"][0][:, 0, roi_lo_ndx:roi_hi_ndx], axis=1)
    return roi_sum


def e_fly_export(db_header):
    """
    Save data in XDI format.
    """

    start = db_header.start
    element = start["user_input"]["element"]
    roi = element_to_roi[element.lower()]
    suitcase_transforms = {"e_fly_roi1": partial(e_fly_roi1, roi_lo_ndx=roi[0], roi_hi_ndx=roi[1])}

    with Serializer(
        directory=os.path.expanduser(
            f"~/Users/Data/{start['operator']}/{datetime.date.today().isoformat()}/e_fly/"
        ),
        file_prefix="{scan_title}-{scan_id}-{operator}-",
        xdi_file_template=suitcase_config,
        transforms=suitcase_transforms,
    ) as serializer:
        for item in db_header.documents(fill=True):
            serializer(*item)

    return serializer.artifacts


def e_fly_serializer_factory(name, start_doc):
    element = start_doc["user_input"]["element"]
    roi = element_to_roi[element.lower()]

    suitcase_transforms = {"e_fly_roi1": partial(e_fly_roi1, roi_lo_ndx=roi[0], roi_hi_ndx=roi[1])}

    serializer = Serializer(
        directory=os.path.expanduser(
            "~/Users/Data/{}".format(datetime.date.today().isoformat())
        ),
        file_prefix="{scan_title}-{scan_id}-",
        xdi_file_template=suitcase_config,
        transforms=suitcase_transforms,
    )
    return [serializer], []

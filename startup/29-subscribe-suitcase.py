import datetime
from functools import partial
import os.path

import numpy as np


suitcase_config = """\
[versions]
"XDI"                         = "# XDI/1.0 Bluesky"

[columns]
"Column.1"                    = {column_label="energy",  data_key="E_centers", column_data="{[data][E_centers][0]}", units="eV"}
"Column.2"                    = {column_label="I0",      data_key="I0", column_data="{data[I0][0]}"}
"Column.3"                    = {column_label="S_Sclr",      data_key="S", column_data="{data[S][0]}"}
"Column.4"                    = {column_label="If_XS",      data_key="fluor", column_data="{data[fluor][0]}", transform="e_fly_roi1"}
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
    suitcase_transforms = {
        "e_fly_roi1": partial(e_fly_roi1, roi_lo_ndx=roi[0], roi_hi_ndx=roi[1])
    }

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

    suitcase_transforms = {
        "e_fly_roi1": partial(e_fly_roi1, roi_lo_ndx=roi[0], roi_hi_ndx=roi[1])
    }

    serializer = Serializer(
        directory=os.path.expanduser(
            "~/Users/Data/{}".format(datetime.date.today().isoformat())
        ),
        file_prefix="{scan_title}-{scan_id}-",
        xdi_file_template=suitcase_config,
        transforms=suitcase_transforms,
    )
    return [serializer], []


def e_step_serializer(element, beamline_operator, suitcase_config):
    """
    Return an XDI serializer with output directory
        "~/Users/Data/{beamline_operator}/{datetime.date.today().isoformat()}/e_step/"
    and file prefix
        "{scan_title}-{scan_id}-"

    Parameters
    ---------
    element: str
    beamline_operator: str
    suitcase_config: str

    Return
    ------
    suitcase.xdi.Serializer
    """
    roi = element_to_roi[element.lower()]
    suitcase_transforms = {
        "e_scan_roi1": partial(e_fly_roi1, roi_lo_ndx=roi[0], roi_hi_ndx=roi[1])
    }
    serializer = Serializer(
        directory=os.path.expanduser(
            f"~/Users/Data/{beamline_operator}/{datetime.date.today().isoformat()}/e_step/"
        ),
        file_prefix="{scan_title}-{scan_id}-",
        xdi_file_template=suitcase_config,
        transforms=suitcase_transforms,
    )
    return serializer


def e_step_export(db_header):
    """
    Export data from DataBroker to XDI format.
    """

    start = db_header.start
    with e_step_serializer(
        element=start["user_input"]["element"],
        beamline_operator=start["operator"],
        suitcase_config=suitcase_config,
    ) as serializer:
        for item in db_header.documents(fill=True):
            serializer(*item)

    return serializer.artifacts


def e_step_serializer_factory(name, start_doc):
    """
    Factory function returning XDI Serializers.

    Parameters
    ----------

    name: str
    start_doc: dict

    Return
    ------

    [serializer], []
    """
    serializer = e_step_serializer(
        element=start_doc["user_input"]["element"],
        beamline_operator=start_doc["operator"],
        suitcase_config=suitcase_config,
    )
    return [serializer], []

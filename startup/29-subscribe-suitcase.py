import numpy as np

from event_model import RunRouter


element_to_roi = {
    "s": (222, 240),
    "p": (222, 240)
}


suitcase_config = """\
[versions]
"XDI"                         = "# XDI/1.0 Bluesky"

[columns]
"Column.1"                    = {column_label="energy",  data_key="E_centers", column_data="{[data][E_centers][0]}", units="eV"}
"Column.2"                    = {column_label="I0",      data_key="I0", column_data="{data[I0][0]}"}
"Column.3"                    = {column_label="If",      data_key="fluor", column_data="{data[fluor][0]}", transform="roi1"}

[required_headers]
"Element.symbol"              = {data="{md[XDI][Element_symbol]}", doc_name="start"}
"Element.edge"                = {data="{md[XDI][Element_edge]}", doc_name="start"}
"Mono.d_spacing"              = {data="{md[XDI][Mono_d_spacing]}", doc_name="start"}

[optional_headers]

"""


def roi1(event_doc, roi_lo_ndx, roi_hi_ndx):
    # doc[data][fluor][0].shape is (243, 1, 4096)
    print(f"doc[data][fluor][0].shape is {event_doc['data']['fluor'][0].shape}")
    roi_sum = np.sum(event_doc["data"]["fluor"][0][:, 0, roi_lo_ndx:roi_hi_ndx], axis=1)
    print(f"roi_sum.shape is {roi_sum.shape}")
    return roi_sum


def serializer_factory(name, start_doc):
    element = start_doc["user_input"]["element"]
    roi = element_to_roi[element.lower()]

    suitcase_transforms = {
        "roi1": partial(roi1, roi_lo_ndx=roi[0], roi_hi_ndx=roi[1])
    }

    serializer = Serializer(directory="xdi", xdi_file_template=suitcase_config, transforms=suitcase_transforms)
    return [serializer], []

##RE.subscribe(RunRouter([serializer_factory]))


def xdi_export(db_header):
    """
    This function is intended for testing on a stream of documents from a databroker.
    """
    s, _ = serializer_factory("start", db_header.start)
    serializer = s[0]
    #with serializer(directory, file_prefix, xdi_file_template=xdi_file_template, transforms=transforms, **kwargs) as serializer:
    for item in db_header.documents(fill=True):
        serializer(*item)
    serializer.close()

    return serializer.artifacts


xdi_meta_data = {"Element_symbol": "???", "Element_edge": "???", "Mono_d_spacing": "???"}

nx_meta_data = {
    "Source": {"name": "NSLS-II"},
    "Instrument": {"name": "TES"},
    "Beam": {"incident_energy": 1000.0},
}

# not like this
#RE.md.update({"md": {"NX": nx_meta_data, "XDI": xdi_meta_data}})

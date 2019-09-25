import numpy as np

from event_model import RunRouter


def roi1(doc):
    return np.sum(doc["data"]["fluor"][0])


suitcase_transforms = {
    "roi1": roi1
}


def serializer_factory(name, start_doc):
    serializer = Serializer("xdi", transforms=suitcase_transforms)
    serializer("start", start_doc)
    return [serializer], []

##RE.subscribe(RunRouter([serializer_factory]))


suitcase_meta_data = {"config": """\
[versions]
"XDI"                         = "# XDI/1.0 Bluesky"

[columns]
"Column.1"                    = {column_label="energy",  data_key="E_centers", column_data="{configuration[E_centers][data][E_centers][0]}", units="eV"}
"Column.2"                    = {column_label="I0",      data_key="I0", column_data="{data[I0][0]}"}
"Column.3"                    = {column_label="If",      data_key="fluor", column_data="{data[fluor][0]}", transform="roi1"}
"Column.4"                    = {column_label="If_",     data_key="xs_channel1_rois_roi01_value", column_data="{data[xs_channel1_rois_roi01_value][0]}"}

[required_headers]
"Element.symbol"              = {data="{md[XDI][Element_symbol]}"}
"Element.edge"                = {data="{md[XDI][Element_edge]}"}
"Mono.d_spacing"              = {data="{md[XDI][Mono_d_spacing]}"}

[optional_headers]

"""}

xdi_meta_data = {"Element_symbol": "???", "Element_edge": "???", "Mono_d_spacing": "???"}

nx_meta_data = {
    "Source": {"name": "NSLS-II"},
    "Instrument": {"name": "BMM"},
    "Beam": {"incident_energy": 1000.0},
}

RE.md.update({"md": {"suitcase-xdi": suitcase_meta_data, "NX": nx_meta_data, "XDI": xdi_meta_data}})

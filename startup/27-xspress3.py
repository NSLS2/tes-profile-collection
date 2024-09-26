print(f"Loading {__file__!r} ...")

import os
from uuid import uuid4
import h5py

from event_model import compose_resource
from ophyd.areadetector import (
    AreaDetector,
    PixiradDetectorCam,
    ImagePlugin,
    TIFFPlugin,
    StatsPlugin,
    HDF5Plugin,
    ProcessPlugin,
    ROIPlugin,
    TransformPlugin,
    OverlayPlugin,
)
from ophyd.areadetector.plugins import PluginBase
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.device import BlueskyInterface
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import (
    FileStoreIterativeWrite,
    FileStoreHDF5IterativeWrite,
    FileStoreTIFFSquashing,
    FileStoreTIFF,
    FileStoreHDF5,
    new_short_uid,
    FileStoreBase,
)
from databroker.assets.handlers import HandlerBase
from ophyd import Kind, Signal
from ophyd import Component as C
from pathlib import Path, PurePath
from nslsii.detectors.xspress3 import (
    XspressTrigger,
    Xspress3Detector,
    Xspress3Channel,
    Xspress3FileStore,
    logger,
)
from enum import Enum
from collections import OrderedDict, deque
from ophyd import Staged
from ophyd.status import DeviceStatus


class TESMode(Enum):
    step = 1
    fly = 2


from ophyd.areadetector.filestore_mixins import FileStorePluginBase


from ophyd import Component
from ophyd.areadetector import Xspress3Detector
from nslsii.areadetector.xspress3 import (
    build_xspress3_class,
    Xspress3HDF5Plugin,
    Xspress3Trigger
)

class TESXspress3DetectorHDF5Plugin(Xspress3HDF5Plugin):


    def __init__(self, *args, md = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._md = md

    def _build_data_dir_path(self, the_datetime, root_path, path_template):
        """
        Construct a data directory path from root_path and path_template.

        Parameters
        ----------
        the_datetime: datetime.datetime
            the date and time to use in formatting path_template
        root_path: str
            the "non-semantic" part of the data path, for example /nsls2/data/tst
        path_template: str
            path to the data directory, which must include the root_path,
            and may include %Y, %m, %d and other strftime replacements,
            for example /nsls2/data/tst/xspress3/%Y/%m/%d
        Return
        ------
          str
        """

        # 1. fill in path_template with the_datetime as AreaDetector would do
        # 2. concatenate result with root_path
        #   if root_path is the prefix of the_data_dir_path
        #   then
        #     Path(root_path) / Path(the_data_dir_path)
        #   will be
        #     Path(the_data_dir_path)
        #   for example, if
        #     root_path         = "/nsls2/data"
        #     the_data_dir_path = "/nsls2/data/tst/xspress3/2020/01/01"
        #   then
        #     the_full_data_dir_path = Path("/nsls2/data/tst/xspress3/2020/01/01")

        beamline = os.getenv("ENDSTATION_ACRONYM", os.getenv("BEAMLINE_ACRONYM", "TST")).lower()
        # These three beamlines have a -new suffix in their 
        if beamline in ["xpd", "fxi", "qas"]:
            beamline = f"{beamline}-new"
        the_full_data_dir_path = f"/nsls2/data/{beamline}/proposals/{self._md.get('cycle', '')}/{self._md.get('data_session', '')}/assets/default"
        the_data_dir_path = the_datetime.strftime(path_template)
        the_full_data_dir_path = Path(the_full_data_dir_path) / Path(the_data_dir_path)
        return the_full_data_dir_path
    

# jlynch: debugging
# class NewXspress3Trigger(Xspress3Trigger):
#     def trigger(self):

#         self.cam.acquire.put(0, wait=True)
#         return super().trigger()

# The Xspress3 Mini at TES has 2 channels but
# as of 2023-05-08 only channel 1 is in use
# so this class only defines channel 1. If the
# second channel is put in use change
#   channel_numbers=(1,)
# to
#   channel_nubmers=(1, 2)
xspress3_class_4ch = build_xspress3_class(
    channel_numbers=(1,2,3,4),
    mcaroi_numbers=(1, 2, 3, 4),
    image_data_key="fluor",
    xspress3_parent_classes=(Xspress3Detector, Xspress3Trigger),
    extra_class_members={
        "hdf5": Component(
            TESXspress3DetectorHDF5Plugin,
            "HDF1:",
            md=RE.md,
            name="hdf5",
            root_path=f"/nsls2/data/tes/proposals/{RE.md.get('cycle', '')}/{RE.md.get('data_session', '')}/assets/",
            path_template="%Y/%m/%d",
        )
    }
)


class TESXspress3Detector4CH(xspress3_class_4ch):
    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method) so we can modify this as part of a plan
    fly_next = Component(Signal, value=False)

    # must set kind on instance
    energy_calibration = Component(Signal, value=10.0, kind="config")

    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None, **kwargs):
        if configuration_attrs is None:
            configuration_attrs = [
                "external_trig",
                "total_points",
                "spectra_per_point",
                "cam",
                "rewindable",
            ]
        if read_attrs is None:
            # E step scan
            # read_attrs = ["channel01", "hdf5"]
            # xy flyscan
            read_attrs = ["fluor", "channel01", "channel02", "channel03", "channel04", "hdf5"]

        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            **kwargs,
        )
        # this is possibly one too many places to store this
        # in the parent class it looks at if the extrenal_trig signal is high
        self._mode = TESMode.step

        self.bulk_data_spec = "XSP3_FLY"

    def stop(self, *, success=False):
        print("Xspress3Detector.stop")
        stop_result = super().stop()
        self.cam.acquire.put(0)
        self.hdf5.stop(success=success)
        return stop_result

    def stage(self):
        # print("starting stage")
        # do the latching
        if self.fly_next.get():
            print("put False to fly_next")
            self.fly_next.put(False)
            self._mode = TESMode.fly

        if self.external_trig.get():
            # print("setting TTL Veto Only trigger mode")
            self.stage_sigs = {
                self.cam.trigger_mode: "TTL Veto Only"
            }
        else:
            # print("setting Internal trigger mode")
            self.stage_sigs = {
                self.cam.trigger_mode: "Internal"
            }

        return super().stage()

    def unstage(self):
        try:
            # when a scan is aborted we are seeing
            # the xspress3 acquire PV remaining at 1
            # we had to use CSS to "stop" the xspress3
            self.cam.acquire.put(0, wait=True)
            unstage_result = super().unstage()
        finally:
            self._mode = TESMode.step
        return unstage_result
    

xspress3_class_1ch = build_xspress3_class(
    channel_numbers=(1,),
    mcaroi_numbers=(1, 2, 3, 4),
    image_data_key="fluor",
    xspress3_parent_classes=(Xspress3Detector, Xspress3Trigger),
    extra_class_members={
        "hdf5": Component(
            TESXspress3DetectorHDF5Plugin,
            "HDF1:",
            md=RE.md,
            name="hdf5",
            root_path=f"/nsls2/data/tes/proposals/{RE.md.get('cycle', '')}/{RE.md.get('data_session', '')}/assets",
            path_template="%Y/%m/%d",
        )
    }
)

class TESXspress3Detector1CH(xspress3_class_1ch):
    # this is used as a latch to put the xspress3 into 'bulk' mode
    # for fly scanning.  Do this is a signal (rather than as a local variable
    # or as a method) so we can modify this as part of a plan
    fly_next = Component(Signal, value=False)

    # must set kind on instance
    energy_calibration = Component(Signal, value=10.0, kind="config")

    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None, **kwargs):
        if configuration_attrs is None:
            configuration_attrs = [
                "external_trig",
                "total_points",
                "spectra_per_point",
                "cam",
                "rewindable",
            ]
        if read_attrs is None:
            # E step scan
            # read_attrs = ["channel01", "hdf5"]
            # xy flyscan
            read_attrs = ["fluor", "channel01", "hdf5"]

        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            **kwargs,
        )
        # this is possibly one too many places to store this
        # in the parent class it looks at if the extrenal_trig signal is high
        self._mode = TESMode.step

        self.bulk_data_spec = "XSP3_FLY"

    def stop(self, *, success=False):
        print("Xspress3Detector.stop")
        stop_result = super().stop()
        self.cam.acquire.put(0)
        self.hdf5.stop(success=success)
        return stop_result

    def stage(self):
        # print("starting stage")
        # do the latching
        if self.fly_next.get():
            print("put False to fly_next")
            self.fly_next.put(False)
            self._mode = TESMode.fly

        if self.external_trig.get():
            # print("setting TTL Veto Only trigger mode")
            self.stage_sigs = {
                self.cam.trigger_mode: "TTL Veto Only"
            }
        else:
            # print("setting Internal trigger mode")
            self.stage_sigs = {
                self.cam.trigger_mode: "Internal"
            }

        return super().stage()

    def unstage(self):
        try:
            # when a scan is aborted we are seeing
            # the xspress3 acquire PV remaining at 1
            # we had to use CSS to "stop" the xspress3
            self.cam.acquire.put(0, wait=True)
            unstage_result = super().unstage()
        finally:
            self._mode = TESMode.step
        return unstage_result
    

XS3_CLASS_MAP = {
    1: TESXspress3Detector1CH,
    4: TESXspress3Detector4CH,
}
    
def tes_xs3_factory(prefix, name, num_channels = 4):
    xs3_class = XS3_CLASS_MAP[num_channels]
    xspress3 = xs3_class(prefix=prefix, name=name)

    # jlynch: SRX does this
    xspress3.hdf5.stage_sigs[xspress3.hdf5.blocking_callbacks] = 1

    xspress3.energy_calibration.kind = "config"

    xspress3.fluor.name = "fluor"
    xspress3.fluor.kind = Kind.normal  # this is for xy flyscan only

    for channel in xspress3.iterate_channels():
        # channel.kind = "normal" this is for e step scan only
        # channel.fluor.shape = (1, 1, 4096)
        for mcaroi in channel.iterate_mcarois():
            # "normal" may be ok as well
            mcaroi.kind = Kind.normal
            mcaroi.total_rbv.kind = Kind.normal

    # Set the first channel's first roi to hinted for BEC
    chan_1_roi_1 = next(next(xspress3.iterate_channels()).iterate_mcarois())
    chan_1_roi_1.kind = Kind.hinted
    chan_1_roi_1.total_rbv.kind = Kind.hinted


    xspress3.cam.configuration_attrs = [
        "acquire_period",
        "acquire_time",
        "image_mode",
        "manufacturer",
        "model",
        "num_exposures",
        "num_images",
        "temperature",
        "temperature_actual",
        "trigger_mode",
        "config_path",
        "config_save_path",
        "invert_f0",
        "invert_veto",
        "xsp_name",
        "num_channels",
        "num_frames_config",
        "run_flags",
        "trigger_signal",
    ]

    return xspress3

xs = tes_xs3_factory(prefix="XF:08BM-ES{Xsp:2}:", name="xs", num_channels=1)
xssmart = tes_xs3_factory(prefix="XF:08BM-ES{XS3:Det-3}:", name="xssmart")


def set_xssmart_roi(element, edge, chanel, *, roi=1, low_bin=None, size=None):

    element = element+"_"+edge
    xssmart_roi_low = EpicsSignal("XF:08BM-ES{XS3:Det-3}:MCA"+str(chanel)+"ROI:"+str(roi)+":MinX", name = "xssmart_roi_low")
    xssmart_roi_high = EpicsSignal("XF:08BM-ES{XS3:Det-3}:MCA"+str(chanel)+"ROI:"+str(roi)+":SizeX", name = "xssmart_roi_high")
    xssmart_roi_name = EpicsSignal("XF:08BM-ES{XS3:Det-3}:MCA"+str(chanel)+"ROI:"+str(roi)+":Name", name = "xssmart_roi_name")
    if low_bin == None or size == None:
        low_bin = element_to_roi_smart[element.lower()][0]
        size = element_to_roi_smart[element.lower()][1]
    xssmart_roi_name.put(element)
    xssmart_roi_low.put(low_bin)
    xssmart_roi_high.put(size)


def set_xs_roi(element, edge, chanel, *, roi = 1, low_bin=None, size=None):

    element = element+"_"+edge
    xs_roi_low = EpicsSignal("XF:08BM-ES{Xsp:2}:MCA"+str(chanel)+"ROI:"+str(roi)+":MinX", name = "xs_roi_low")
    xs_roi_high = EpicsSignal("XF:08BM-ES{Xsp:2}:MCA"+str(chanel)+"ROI:"+str(roi)+":SizeX", name = "xs_roi_high")
    xs_roi_name = EpicsSignal("XF:08BM-ES{Xsp:2}:MCA"+str(chanel)+"ROI:"+str(roi)+":Name", name = "xs_roi_name")
    if low_bin == None or size == None:
        low_bin = element_to_roi[element.lower()][0]
        size = element_to_roi[element.lower()][1]

    xs_roi_name.put(element)
    xs_roi_low.put(low_bin)
    xs_roi_high.put(size)

# is this necessary?
# xs.channel1.rois.read_attrs = ["roi{:02}".format(j) for j in [1, 2, 3, 4]]
# xs.hdf5.num_extra_dims.put(0)
# xs.channel1.vis_enabled.put(1)
# xs.cam.num_channels.put(1)




class BulkXspress3Handler(HandlerBase):
    HANDLER_NAME = "BULK_XSPRESS3"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self, frame=-1, channel=-1):
        if channel < 0:
            return self._handle["entry/instrument/detector/data"][:]
        else:
            return self._handle["entry/instrument/detector/data"][:, channel, :]

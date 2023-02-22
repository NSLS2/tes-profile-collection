import logging
import os
from pathlib import Path

import appdirs
from bluesky.utils import PersistentDict
from IPython import get_ipython
from nslsii import configure_base
from ophyd.signal import EpicsSignalBase

EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)

configure_base(get_ipython().user_ns,
               "tes",
               publish_documents_with_kafka=True)


# runengine_metadata_dir = appdirs.user_data_dir(appname="bluesky") / Path("runengine-metadata")
# runengine_metadata_dir = Path('/nsls2/xf08bm/shared/config/runengine-metadata')
runengine_metadata_dir = Path("/nsls2/data/tes/shared/config/runengine-metadata")

# PersistentDict will create the directory if it does not exist
RE.md = PersistentDict(runengine_metadata_dir)


# Optional: set any metadata that rarely changes.
RE.md["beamline_id"] = "TES"


def warmup_hdf5_plugins(detectors):
    """
    Warm-up the hdf5 plugins.

    This is necessary for when the corresponding IOC restarts we have to trigger one image
    for the hdf5 plugin to work correctly, else we get file writing errors.

    Parameter:
    ----------
    detectors: list
    """
    for det in detectors:
        _array_size = det.hdf5.array_size.get()
        if 0 in [_array_size.height, _array_size.width] and hasattr(det, "hdf5"):
            print(f"\n  Warming up HDF5 plugin for {det.name} as the array_size={_array_size}...")
            det.hdf5.warmup()
            print(f"  Warming up HDF5 plugin for {det.name} is done. array_size={det.hdf5.array_size.get()}\n")
        else:
            print(f"\n  Warming up of the HDF5 plugin is not needed for {det.name} as the array_size={_array_size}.")


def auto_alignment_mode(envvar="AUTOALIGNMENT", default="no"):
    """Enable auto-alignment mode.

    In that mode the `bloptools` library will be imported and some suspenders will be disabled.

    Returns
    -------
        True if the auto-alignment mode is enabled, False otherwise.
    """
    if os.getenv(envvar, default).lower() in ["yes", "y", "true", "1"]:
        return True
    else:
        return False

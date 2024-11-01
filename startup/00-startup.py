print(f"Loading {__file__!r} ...")

import logging
import os
from pathlib import Path
import time
from tiled.client import from_profile

import appdirs
from bluesky.utils import PersistentDict
from IPython import get_ipython
from nslsii import configure_base, configure_kafka_publisher
from ophyd.signal import EpicsSignalBase
import redis
from redis_json_dict import RedisJSONDict
from IPython.terminal.prompts import Prompts, Token


EpicsSignalBase.set_defaults(timeout=10, connection_timeout=10)

class ProposalIDPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        return [
            (
                Token.Prompt,
                f"{RE.md.get('data_session', 'N/A')} [",
            ),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]


ip = get_ipython()
ip.prompts = ProposalIDPrompt(ip)


class TiledInserter:
    def insert(self, name, doc):
        ATTEMPTS = 20
        error = None
        for attempt in range(ATTEMPTS):
            try:
                tiled_writing_client.post_document(name, doc)
            except Exception as exc:
                print("Document saving failure:", repr(exc))
                error = exc
            else:
                break
            time.sleep(2)
        else:
            # Out of attempts
            raise error

tiled_inserter = TiledInserter()

configure_base(get_ipython().user_ns,
               tiled_inserter,
               publish_documents_with_kafka=False)

# TODO: This is a workaround, remove this once nslsii is fixed. See: https://github.com/NSLS-II/nslsii/issues/207
configure_kafka_publisher(RE, "tes")

#RE.unsubscribe(0)  # Remove old-style databroker saving.# Define tiled catalog
tiled_writing_client = from_profile("nsls2", api_key=os.environ["TILED_BLUESKY_WRITING_API_KEY_TES"])["tes"]["raw"]

print("Intializing tiled reading client...\nMake sure you check for duo push.")
db = tiled_reading_client = from_profile("nsls2")["tes"]["raw"]


def whoami():
    try:
        print(f"\nLogged in to tiled as: {db.context.whoami()['identities'][0]['id']}\n")
    except TypeError as e:
        print("Not authenticated with tiled! Please login...")

whoami()

def login():
    db.login()

def logout():
    """
    Logout of tiled and reset the default username.
    This is needed to switch between different users.
    """

    tiled_reading_client.logout()
    from tiled.client.context import clear_default_identity
    clear_default_identity(tiled_reading_client.context.api_uri)

#RE.subscribe(post_document)


# runengine_metadata_dir = appdirs.user_data_dir(appname="bluesky") / Path("runengine-metadata")
# runengine_metadata_dir = Path('/nsls2/xf08bm/shared/config/runengine-metadata')
runengine_metadata_dir = Path("/nsls2/data/tes/shared/config/runengine-metadata")

RE.md = RedisJSONDict(redis.Redis("info.tes.nsls2.bnl.gov"), prefix="")

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

    In that mode, the `bloptools` library will be imported and some suspenders will be disabled.

    Returns
    -------
        True if the auto-alignment mode is enabled, False otherwise.
    """
    if os.getenv(envvar, default).lower() in ["yes", "y", "true", "1"]:
        return True
    else:
        return False


def print_doc_to_stdout(name, doc):

    print(f"\n================= Emitting {name} document =====================\n")
    print(f"{doc}")
    print(f"\n========================== Done ================================\n")


# Uncomment to view raw document stream
#RE.subscribe(print_doc_to_stdout)

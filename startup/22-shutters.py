print(f"Loading {__file__!r} ...")

from nslsii.devices import TwoButtonShutter


# TODO: This shutter class needs some work to account for the cases of multiple
# and prolonged state switches.
#
#   https://github.com/NSLS-II-TES/profile_collection/issues/34
#
# psh = TwoButtonShutter("XF:08BMES-PPS{PSh}", name="psh")
# Usage:
#   psh.set("Open").wait()
#   psh.set("Close").wait()


class PShutter(Device):
    # TODO: extend the class to take care of the flaky state switching.
    open_cmd = Cpt(EpicsSignal, "Cmd:Opn-Cmd")
    close_cmd = Cpt(EpicsSignal, "Cmd:Cls-Cmd")
    status = Cpt(EpicsSignalRO, "Pos-Sts")


psh = PShutter("XF:08BMES-PPS{PSh}", name="psh")

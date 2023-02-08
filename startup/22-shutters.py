from nslsii.devices import TwoButtonShutter

#psh = TwoButtonShutter("XF:08BMES-PPS{PSh}", name="psh")

class PShutter(Device):
    open_cmd = Cpt(EpicsSignal, "Cmd:Opn-Cmd")
    close_cmd = Cpt(EpicsSignal, "Cmd:Cls-Cmd")
    status = Cpt(EpicsSignalRO, "Pos-Sts")
    
psh = PShutter("XF:08BMES-PPS{PSh}", name="psh")


# psh.set("Open").wait()
# psh.set("Close").wait()

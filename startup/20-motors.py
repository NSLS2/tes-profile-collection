print(f"Loading {__file__!r} ...")

from ophyd import EpicsSignal, EpicsSignalRO, EpicsMotor, Device, Component as Cpt


class MresMotor(EpicsMotor):
    mres = Cpt(EpicsSignal, ".MRES", kind="config")


class XYStage(Device):
    x = Cpt(MresMotor, "X}Mtr")
    y = Cpt(MresMotor, "Y}Mtr")
    z = Cpt(EpicsMotor, "Z}Mtr")


xy_stage = XYStage("XF:08BMES-OP{SM:1-Ax:", name="xy_stage")

xy_stage.x.velocity.kind = "normal"


class Mono(Device):
    # pseudo axis and configuration
    energy = Cpt(EpicsMotor, "x08bm:mon")
    cal = Cpt(EpicsSignal, "x08bm:mono_cal")
    e_back = Cpt(EpicsSignal, "x08bm:E_back")

    # linear drive
    linear = Cpt(MresMotor, "XF:08BMA-OP{Mono:1-Ax:Linear}Mtr")
    tilt = Cpt(MresMotor, "XF:08BMA-OP{Mono:1-Ax:Tilt}Mtr")


mono = Mono(name="mono")
#mono.energy.settle_time = 3
mono.linear.settle_time = 2
#mono.settle_time = 1

dtt = EpicsSignal("XF:08BM-CT{MC:06}Asyn.AOUT", name="dtt", string=True)


class ToroidalMirror(Device):
    dsy = Cpt(EpicsMotor, "XD}Mtr")
    usy = Cpt(EpicsMotor, "XU}Mtr")
    dsh = Cpt(EpicsMotor, "YD}Mtr")  # Should move high to low
    ush = Cpt(EpicsMotor, "YU}Mtr")  # Should move high to low


toroidal_mirror = ToroidalMirror("XF:08BMA-OP{Mir:FM-Ax:", name="toroidal_mirror")
#toroidal_mirror.kind = "hinted"
#toroidal_mirror.dsy.kind = "hinted"
#toroidal_mirror.usy.kind = "hinted"
#toroidal_mirror.dsh.kind = "hinted"
#toroidal_mirror.ush.kind = "hinted"


class SSA(Device):
    inboard = Cpt(EpicsMotor, "I}Mtr")
    outboard = Cpt(EpicsMotor, "O}Mtr")


ssa = SSA("XF:08BMES-OP{SSA:1-Ax:", name="ssa")
ssa.kind = "hinted"
ssa.inboard.kind = "hinted"
ssa.outboard.kind = "hinted"


sd.baseline = [
    toroidal_mirror.dsy,
    toroidal_mirror.usy,
    toroidal_mirror.dsh,
    toroidal_mirror.ush,
    ssa.inboard,
    ssa.outboard,
]


class KBMirror(Device):
    dsh = Cpt(EpicsMotor, "BD}Mtr")
    ush = Cpt(EpicsMotor, "YD}Mtr")

    # BL staff does not want to expose these EpicsMotors PVs via ophyd/bluesky as they are manually controlled via CSS,
    # therefore we add individual components to read the values and record them as configuration attrs:
    dsb_rbv = Cpt(EpicsSignalRO, "BU}Mtr.RBV", kind="config")
    dsb = Cpt(EpicsSignalRO, "BU}Mtr.VAL", kind="config")
    usb_rbv = Cpt(EpicsSignalRO, "YU}Mtr.RBV", kind="config")
    usb = Cpt(EpicsSignalRO, "YU}Mtr.VAL", kind="config")

kbh = KBMirror("XF:08BMES-OP{Mir:KBH-Ax:", name="kbh")
kbv = KBMirror("XF:08BMES-OP{Mir:KBV-Ax:", name="kbv")


class Robot_Smart(Device):
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")
    z = Cpt(EpicsMotor, "Z}Mtr")
    ry = Cpt(EpicsMotor, "Ry}Mtr")

robot_smart = Robot_Smart("XF:08BMC-ES:SE{SmplM:1-Ax:", name = "robot_smart")


robot_x_home = EpicsSignal("XF:08BMC-ES:SE{SmplM:1-Ax:X}Sts:HomeCmplt-Sts", name="robot_x_home")
robot_y_home = EpicsSignal("XF:08BMC-ES:SE{SmplM:1-Ax:Y}Sts:HomeCmplt-Sts", name="robot_y_home")
robot_z_home = EpicsSignal("XF:08BMC-ES:SE{SmplM:1-Ax:Z}Sts:HomeCmplt-Sts", name="robot_z_home")
robot_ry_home = EpicsSignal("XF:08BMC-ES:SE{SmplM:1-Ax:Ry}Sts:HomeCmplt-Sts", name="robot_ry_home")

class Sample_Smart(Device):
    x = Cpt(EpicsMotor, "X}Mtr")
    y = Cpt(EpicsMotor, "Y}Mtr")
    z = Cpt(EpicsMotor, "Z}Mtr")
    ry = Cpt(EpicsMotor, "Ry}Mtr")

sample_smart = Sample_Smart("XF:08BMC-ES:SE{Smpl:1-Ax:", name = "sample_smart")
sample_x_home = EpicsSignal("XF:08BMC-ES:SE{Smpl:1-Ax:X}Sts:HomeCmplt-Sts", name="sample_x_home")
sample_y_home = EpicsSignal("XF:08BMC-ES:SE{Smpl:1-Ax:Y}Sts:HomeCmplt-Sts", name="sample_y_home")
sample_z_home = EpicsSignal("XF:08BMC-ES:SE{Smpl:1-Ax:Z}Sts:HomeCmplt-Sts", name="sample_z_home")
sample_ry_home = EpicsSignal("XF:08BMC-ES:SE{Smpl:1-Ax:Ry}Sts:HomeCmplt-Sts", name="sample_ry_home")

class SDD_Smart(Device):
    x = Cpt(EpicsMotor, "X}Mtr")
SDD_smart = SDD_Smart("XF:08BMC-ES:SE{Det:1-Ax:", name = "SDD_smart")
SDD_smart_home = EpicsSignal("XF:08BMC-ES:SE{Det:1-Ax:X}Sts:HomeCmplt-Sts", name="sample_ry_home")



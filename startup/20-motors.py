from ophyd import (EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Component as Cpt, Signal)

class MresMotor(EpicsMotor):
    mres = Cpt(EpicsSignal, '.MRES', kind='config')

class XYStage(Device):
    x = Cpt(MresMotor, 'X}Mtr')
    y = Cpt(MresMotor, 'Y}Mtr')
    z = Cpt(EpicsMotor, 'Z}Mtr')

xy_stage = XYStage('XF:08BMES-OP{SM:1-Ax:', name='xy_stage')

xy_stage.x.velocity.kind = 'normal'

mono_energy = EpicsMotor('x08bm:mon', name='mono_energy')
mono_energy.settle_time = 1.0

dtt = EpicsSignal('XF:08BM-CT{MC:06}Asyn.AOUT', name='dtt',
                  string=True)


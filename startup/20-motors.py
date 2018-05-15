from ophyd import (EpicsSignal, EpicsSignalRO, EpicsMotor,
                   Device, Component as Cpt, Signal)


class XYStage(Device):
    x = Cpt(EpicsMotor, 'X}Mtr')
    y = Cpt(EpicsMotor, 'Y}Mtr')


xy_stage = XYStage('XF:08BMES-OP{SM:1-Ax:', name='xy_stage')

mono_energy = EpicsMotor('x08bm:mon', name='mono_energy')
mono_energy.settle_time = 1.0

dtt = EpicsSignal('XF:08BM-CT{MC:06}Asyn.AOUT', name='dtt',
                  as_string=True)

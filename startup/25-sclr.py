import time
from ophyd import EpicsScaler, EpicsSignal, EpicsSignalRO

class DirtyLyingEpicsScaler(EpicsScaler):
    def trigger(self):
        ret = super().trigger()
        # The .CNT signal seems to fire before the device is actually up to date
        # and thus we read a value that is slightly too low! This fixes it.
        time.sleep(0.1)
        return ret

sclr = DirtyLyingEpicsScaler('XF:08BM-ES:1{Sclr:1}scaler1', name='sclr')
sclr.channels.read_attrs = ['chan1', 'chan2', 'chan3', 'chan4', 'chan5', 'chan6', 'chan7', 'chan8', 'chan9', 'chan10', 'chan11', 'chan12', 'chan13', 'chan14', 'chan15', 'chan16', 'chan17', 'chan18', 'chan19', 'chan20']
sclr.channels.chan1.name = 'Clock'
sclr.channels.chan2.name = 'I0'
sclr.channels.chan3.name = 'PIN'
sclr.channels.chan4.name = 'VxCl'
sclr.channels.chan5.name = 'fbratio'
sclr.channels.chan6.name = 'It'
sclr.channels.chan7.name = 'If_tot'
sclr.channels.chan8.name = 'S'
sclr.channels.chan9.name = 'Mg'
sclr.channels.chan10.name = 'P'
sclr.channels.chan11.name = 'VxP'
sclr.channels.chan12.name = 'Sr_Si'
sclr.channels.chan13.name = 'Ca'
sclr.channels.chan14.name = 'U'
sclr.channels.chan15.name = 'VxAl'
sclr.channels.chan16.name = 'VxMg'
sclr.channels.chan17.name = 'VxS'
sclr.channels.chan18.name = 'VxSi'
sclr.channels.chan19.name = 'VxPu'
sclr.channels.chan20.name = 'Cl'

sclr.count_mode.set('OneShot')  # make sure we are in the mode we expect

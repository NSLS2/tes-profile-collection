from ophyd import EpicsScaler, EpicsSignal, EpicsSignalRO

sclr = EpicsScaler('XF:08BM-ES:1{Sclr:1}scaler1', name='sclr')
sclr.channels.read_attrs = ['chan2', 'chan3']
sclr.channels.chan2.name = 'I0'
sclr.channels.chan3.name = 'PIN'

from ophyd.scaler import ScalerCH
from ophyd.device import (Component as C, DynamicDeviceComponent as DDC,
                          FormattedComponent as FC)
from ophyd.status import StatusBase


class ScalerMCA(Device):
    _default_read_attrs = ('channels', 'current_channel')
    _default_configuration_attrs = ('nuse', 'prescale')
    
    channels = DDC({f'mca{k:02d}': (EpicsSignal, f"mca{k}", {}) for k in range(1, 21)})
    startall = C(EpicsSignal, 'StartAll', string=True)
    stopall = C(EpicsSignal, 'StopAll', string=True)
    eraseall = C(EpicsSignal, 'EraseAll', string=True)
    erasestart = C(EpicsSignal, 'EraseStart', string=True)

    current_channel = C(EpicsSignal, 'CurrentChannel')
    nuse = C(EpicsSignal, 'NuseAll')
    prescale = C(EpicsSignal, 'Prescale')

    # high is acquiring
    status = C(EpicsSignal, 'Acquiring', string=True)

    def stage(self):
        super().stage()
        self.eraseall.put('Erase')

    def stop(self):
        self.stopall.put('Stop')

    def trigger(self):
        self.erasestart.put('Erase')

        return StatusBase()
    

class Scaler(Device):
    # MCAs
    mcas = C(ScalerMCA, '')
    # TODO maybe an issue with the timing around the triggering?
    cnts = C(ScalerCH, 'scaler1')
    
    def __init__(self, *args, mode='counting', **kwargs):
        super().__init__(*args, **kwargs)
        self.set_mode(mode)


    def match_names(self, N=20):
        self.cnts.match_names()
        for j in range(1, N+1):
            mca_ch = getattr(self.mcas.channels, f"mca{j:02d}")
            ct_ch = getattr(self.cnts.channels, f"chan{j:02d}")
            mca_ch.name = ct_ch.chname.get()

    # TODO put a soft signal around this so we can stage it
    def set_mode(self, mode):
        if mode == 'counting':
            self.read_attrs = ['cnts']
            self.configuration_attrs = ['cnts']
        elif mode == 'flying':
            self.read_attrs = ['mcas']
            self.configuration_attrs = ['mcas']
        else:
            raise ValueError

        self._mode = mode

    def trigger(self):
        if self._mode == 'counting':
            return self.cnts.trigger()
        elif mode == 'flying':
            return self.mcas.trigger()
        else:
            raise ValueError
        
    def stage(self):
        self.match_names()
        if self._mode == 'counting':
            return self.cnts.stage()
        elif mode == 'flying':
            return self.mcas.stage()
        else:
            raise ValueError

    def unstage(self):
        if self._mode == 'counting':
            return self.cnts.unstage()
        elif mode == 'flying':
            return self.mcas.unstage()
        else:
            raise ValueError


sclr = Scaler('XF:08BM-ES:1{Sclr:1}', name='sclr')
sclr.cnts.channels.read_attrs = [f"chan{j:02d}" for j in range(1, 21)]
sclr.mcas.channels.read_attrs = [f"mca{j:02d}" for j in range(1, 21)]
sclr.match_names(20)
sclr.set_mode('counting')


def tes_fly_struck(xstart, xstop, ystart, ystop, ysteps):
    sclr.set_mode('flying')

    for y in np.linspace(ystart, ystop, ysteps):
        yield from bps.mv(x_mtr, x_start, y_mtr, y)
        yield from bps.trigger(sclr, group='fly_row'))
        yield from bps.abs_set(x_mtr, xstop, group='fly_row')

        yield from bps.trigger_and_read([sclr])
        

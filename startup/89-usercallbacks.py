import tifffile
from bluesky.callbacks import CallbackBase
import numpy as np


class RasterMaker(CallbackBase):
    def __init__(self, fname_template, field):
        self.fname_template = fname_template
        self.field = field
        self.data = None
        self.desc = None
        self.fname = None

    def start(self, doc):
        if 'shape' in doc:
            self.data = np.ones(doc['shape']) * np.nan
            self.fname = self.fname_template.format(start=doc)

    def descriptor(self, doc):
        if self.field in doc['data_keys']:
            self.decs = doc['uid']

    def event(self, doc):
        if self.desc != doc['descriptor']:
            return
        indx = np.unravel_index(doc['seq_num'], self.shape)
        self.data[indx] = doc['data'][self.field]

    def stop(self, doc):
        tifffile.imsave(self.fname, self.data)
        self.data = None
        self.desc = None
        self.fname = None

from itertools import count as _it_count
import matplotlib.pyplot as plt

class EScanPlot(CallbackBase):
    def __init__(self):
        self.known_desc = {}
        self.index = _it_count()
        self.fig = self.ax = None
        self.energy_bins = None
        self._current_data = None
        self.current_line = None

    def start(self, doc):
        self.fig = plt.figure(f"scan: {doc['scan_id']}")
        self.ax = self.fig.gca()
        self.ax.set_xlabel('Mono Energy (eV)')
        self.ax.set_ylabel('ROI1')
        self.energy_bins = None
        self._current_data = None
        self.current_line = None

    def descriptor(self, doc):
        print(doc['name'])
        if doc['name'] in ('row_ends', 'energy_bins', 'xs_channel1_rois_roi01_value_monitor'):
            self.known_desc[doc['uid']] = doc

    def event(self, doc):
        desc = self.known_desc.get(doc['descriptor'], None)
        if desc is None:
            return

        if desc['name'] == 'row_ends':
            self.index = _it_count()
            return

        if desc['name'] == 'energy_bins':
            self.energy_bins = doc['data']['E_centers']
            self.ax.set_xlim(self.energy_bins[0], self.energy_bins[-1])
            return

        if desc['name'] == 'xs_channel1_rois_roi01_value_monitor':
            if self.energy_bins is None:
                return
            bin = next(self.index)
            if bin == 0:
                self._current_data = np.ones_like(self.energy_bins) * np.nan

                self.current_line, = self.ax.plot(self.energy_bins, self._current_data)
            self._current_data[bin] = doc['data']['xs_channel1_rois_roi01_value']
            self.current_line.set_ydata(self._current_data)
            self.ax.relim()
            self.ax.autoscale(scalex=False)
            self.fig.canvas.draw_idle()





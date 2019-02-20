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

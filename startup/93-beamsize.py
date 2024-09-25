
print(f"Loading {__file__!r} ...")

from bluesky.plans import list_scan
from bluesky.plan_stubs import abs_set

from ophyd.utils import LimitError
from ophyd import Signal
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from event_model import RunRouter
import numpy as np
import pandas as pd

xy_fly_stage = xy_stage

def X_Step_Scan(scan_title, *, operator, element, dwell_time=3, x_sections, step_size, num_scans, xspress3=None):
#def E_Step_Scan(dwell_time,*, scan_title = "abc",E_sections = [2700, 2800, 2900, 3200], step_size = [4, 1, 2], num_scans=2, element = 's'):


    #for v in ["p1600=0", "p1607=4", "p1601=5", "p1602 = 2", "p1600=1"]:
        #yield from bps.mv(dtt, v)
        #yield from bps.sleep(0.1)
    roi = rois(element)
#    yield from bps.mv(xs.channel1.rois.roi01.bin_low, roi[0],
#                  xs.channel1.rois.roi01.bin_high, roi[1])
#    yield from bps.sleep(0.1)
#    xs.channel1.rois.roi01.bin_low.set(roi[0])
#    xs.channel1.rois.roi01.bin_high.set(roi[1])
    x_sections = np.array(x_sections)
    step_size = np.array(step_size)

    ept = []
    for ii in range(step_size.shape[0]):
        ept = ept[0:-1]
        ept = np.append(ept, np.linspace(x_sections[ii], x_sections[ii+1], np.int((x_sections[ii+1] - x_sections[ii])/step_size[ii])+1))
#        print(ept)
    sclr.set_mode("counting")
    yield from bps.mv(xs.external_trig, False) # xs triger mode false means internal trigger
    yield from bps.mv(sclr.cnts.preset_time, dwell_time,
                      xs.settings.acquire_time, dwell_time)#setting dwell time

    #yield from bps.mv(sclr.set_mode,"counting")

    #yield from bps.sleep(0.1)
    #@bpp.monitor_during_decorator([xs.channel1.rois.roi01.value])
    #@bpp.baseline_decorator([mono, xy_stage])
    # TODO put in other meta data
    def scan_once():
        #yield from bps.mv(mono.linear.velocity, 0.5)
        return (yield from list_scan(
            [sclr, xs],
            xy_stage.x,
            ept,
            md={
                "scan_title": scan_title,
                "operator": operator,
                "element": element,
                "user_input": {
                    "element": element,
                    "x_sections": x_sections,
                    "dwell_time": dwell_time,
                    "step_size": step_size,
                },
                "derived_input": {
                    "x":xy_stage.x.position,
                    "y":xy_stage.y.position,
                    "z":xy_stage.z.position,
                   # "Monochromator": Si(111)
                },
            }

        ))
        #yield from bps.mv(mono.linear.velocity, 0.1)


    for scan_iter in range(num_scans):
        yield from scan_once()
        export_X_step(-1,scan_iter)



def export_X_step(scanID=-1, scan_iter=0):
    h = db[scanID] # read data from databroker
    x = h.table()['xy_stage_x']
    I0 = h.table()['I0']
    I_TEY = h.table()['fbratio']
    If_1_roi1 = h.table()['xs_channel1_rois_roi01_value_sum']
    If_1_roi2 = h.table()['xs_channel1_rois_roi02_value_sum']
    If_1_roi3 = h.table()['xs_channel1_rois_roi03_value_sum']
    If_1_roi4 = h.table()['xs_channel1_rois_roi04_value_sum']
    If_2_roi1 = h.table()['xs_channel2_rois_roi01_value_sum']
    If_2_roi2 = h.table()['xs_channel2_rois_roi02_value_sum']
    If_2_roi3 = h.table()['xs_channel2_rois_roi03_value_sum']
    If_2_roi4 = h.table()['xs_channel2_rois_roi04_value_sum']

    df = pd.DataFrame({'#X_position': x, 'I0': I0, 'If_CH1_roi1': If_1_roi1})
    start = h.start
    dt = datetime.datetime.fromtimestamp(start["time"])

    file_head ={'this is a test scan'
                }

    filepath = os.path.expanduser(
        f"~/Users/Data/{start['operator']}/{dt.date().isoformat()}/X_step/"
        f"{start['scan_title']}-{start['scan_id']}-{start['operator']}-{dt.time().strftime('%H-%M-%S')}-{scan_iter}.cvs")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wt") as output_file:
        output_file.write(pprint.pformat(file_head,width=100))
        output_file.write('\n')
        output_file.write('\n')
        output_file.write('\n')

    df.to_csv(filepath, header = True,index = False,mode='a')
    print(f'Data exported to {filepath}')




def Y_Step_Scan(scan_title, *, operator, element, dwell_time=3, y_sections, step_size, num_scans, xspress3=None):
#def E_Step_Scan(dwell_time,*, scan_title = "abc",E_sections = [2700, 2800, 2900, 3200], step_size = [4, 1, 2], num_scans=2, element = 's'):


    #for v in ["p1600=0", "p1607=4", "p1601=5", "p1602 = 2", "p1600=1"]:
        #yield from bps.mv(dtt, v)
        #yield from bps.sleep(0.1)
    roi = rois(element)
#    yield from bps.mv(xs.channel1.rois.roi01.bin_low, roi[0],
#                  xs.channel1.rois.roi01.bin_high, roi[1])
#    yield from bps.sleep(0.1)
#    xs.channel1.rois.roi01.bin_low.set(roi[0])
#    xs.channel1.rois.roi01.bin_high.set(roi[1])
    y_sections = np.array(y_sections)
    step_size = np.array(step_size)

    ept = []
    for ii in range(step_size.shape[0]):
        ept = ept[0:-1]
        ept = np.append(ept, np.linspace(y_sections[ii], y_sections[ii+1], np.int((y_sections[ii+1] - y_sections[ii])/step_size[ii])+1))
#        print(ept)
    sclr.set_mode("counting")
    yield from bps.mv(xs.external_trig, False) # xs triger mode false means internal trigger
    yield from bps.mv(sclr.cnts.preset_time, dwell_time,
                      xs.settings.acquire_time, dwell_time)#setting dwell time

    #yield from bps.mv(sclr.set_mode,"counting")

    #yield from bps.sleep(0.1)
    #@bpp.monitor_during_decorator([xs.channel1.rois.roi01.value])
    #@bpp.baseline_decorator([mono, xy_stage])
    # TODO put in other meta data
    def scan_once():
        #yield from bps.mv(mono.linear.velocity, 0.5)
        return (yield from list_scan(
            [sclr, xs],
            xy_stage.y,
            ept,
            md={
                "scan_title": scan_title,
                "operator": operator,
                "element": element,
                "user_input": {
                    "element": element,
                    "y_sections": y_sections,
                    "dwell_time": dwell_time,
                    "step_size": step_size,
                },
                "derived_input": {
                    "x":xy_stage.x.position,
                    "y":xy_stage.y.position,
                    "z":xy_stage.z.position,
                   # "Monochromator": Si(111)
                },
            }

        ))
        #yield from bps.mv(mono.linear.velocity, 0.1)


    for scan_iter in range(num_scans):
        yield from scan_once()
        export_Y_step(-1,scan_iter)



def export_Y_step(scanID=-1, scan_iter=0):
    h = db[scanID] # read data from databroker
    y = h.table()['xy_stage_y']
    I0 = h.table()['I0']
    I_TEY = h.table()['fbratio']
    If_1_roi1 = h.table()['xs_channel1_rois_roi01_value_sum']
    If_1_roi2 = h.table()['xs_channel1_rois_roi02_value_sum']
    If_1_roi3 = h.table()['xs_channel1_rois_roi03_value_sum']
    If_1_roi4 = h.table()['xs_channel1_rois_roi04_value_sum']
    If_2_roi1 = h.table()['xs_channel2_rois_roi01_value_sum']
    If_2_roi2 = h.table()['xs_channel2_rois_roi02_value_sum']
    If_2_roi3 = h.table()['xs_channel2_rois_roi03_value_sum']
    If_2_roi4 = h.table()['xs_channel2_rois_roi04_value_sum']

    df = pd.DataFrame({'#Y_position': y, 'I0': I0, 'If_CH1_roi1': If_1_roi1})
    start = h.start
    dt = datetime.datetime.fromtimestamp(start["time"])

    file_head ={'this is a test scan'
                }

    filepath = os.path.expanduser(
        f"~/Users/Data/{start['operator']}/{dt.date().isoformat()}/Y_step/"
        f"{start['scan_title']}-{start['scan_id']}-{start['operator']}-{dt.time().strftime('%H-%M-%S')}-{scan_iter}.cvs")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wt") as output_file:
        output_file.write(pprint.pformat(file_head,width=100))
        output_file.write('\n')
        output_file.write('\n')
        output_file.write('\n')

    df.to_csv(filepath, header = True,index = False,mode='a')
    print(f'Data exported to {filepath}')
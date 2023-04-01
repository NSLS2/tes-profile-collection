import matplotlib.pyplot as plt
import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.callbacks import LiveTable, LivePlot


from bluesky.plans import list_scan
import bluesky.plans as bp
from bluesky.plan_stubs import mv

# The line above causes issues with bluesky-queueserver:
# from bluesky.plan_stubs import one_1d_step

# [E 2023-04-01 17:25:38,999 bluesky_queueserver.manager.worker] Failed to start RE Worker environment. Error while loading startup code: Failed to create description of plan 'one_1d_step': Parameter 'take_reading': The expression (<function trigger_and_read at 0x7ff0ae7ab640>) can not be evaluated with 'ast.literal_eval()': unsupported type of default value..
# Traceback (most recent call last):
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2582, in convert_expression_to_string
#     ast.literal_eval(s_value)
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/ast.py", line 64, in literal_eval
#     node_or_string = parse(node_or_string.lstrip(" \t"), mode='eval')
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/ast.py", line 50, in parse
#     return compile(source, filename, mode, flags,
#   File "<unknown>", line 1
#     <function trigger_and_read at 0x7ff0ae7ab640>
#     ^
# SyntaxError: invalid syntax

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2717, in _process_plan
#     default = convert_expression_to_string(p.default, expression_role="default value")
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2586, in convert_expression_to_string
#     raise ValueError(
# ValueError: The expression (<function trigger_and_read at 0x7ff0ae7ab640>) can not be evaluated with 'ast.literal_eval()': unsupported type of default value.

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2719, in _process_plan
#     raise ValueError(f"Parameter '{p.name}': {ex}")
# ValueError: Parameter 'take_reading': The expression (<function trigger_and_read at 0x7ff0ae7ab640>) can not be evaluated with 'ast.literal_eval()': unsupported type of default value.

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/worker.py", line 1130, in run
#     epd = existing_plans_and_devices_from_nspace(nspace=self._re_namespace)
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2900, in existing_plans_and_devices_from_nspace
#     existing_plans = _prepare_plans(plans_in_nspace, existing_devices=existing_devices)
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2786, in _prepare_plans
#     return {
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2787, in <dictcomp>
#     k: _process_plan(v, existing_devices=existing_devices, existing_plans=plan_names) for k, v in plans.items()
#   File "/nsls2/conda/envs/2023-1.3-py310-tiled/lib/python3.10/site-packages/bluesky_queueserver/manager/profile_ops.py", line 2764, in _process_plan
#     raise ValueError(f"Failed to create description of plan '{plan.__name__}': {ex}") from ex
# ValueError: Failed to create description of plan 'one_1d_step': Parameter 'take_reading': The expression (<function trigger_and_read at 0x7ff0ae7ab640>) can not be evaluated with 'ast.literal_eval()': unsupported type of default value.


from bluesky.preprocessors import finalize_wrapper
from bluesky.preprocessors import subs_wrapper
from bluesky.utils import short_uid as _short_uid

# import scanoutput
import numpy
import time
from epics import PV
import collections


def escan():
    """
    Scan the mono_energy while reading the scaler.

    Parameters
    ----------
    start : number
    stop : number
    num : integer
        number of data points (i.e. number of strides + 1)
    md : dictionary, optional
    """

    """
    dets = [xs]
    motor = mono.energy
    cols = ['I0', 'fbratio', 'It', 'If_tot']
    x = 'mono_energy'
    fig, axes = plt.subplots(2, sharex=True)
    plan = bp.scan(dets, motor, start, stop, num, md=md)
    plan2 = bpp.subs_wrapper(plan, [LiveTable(cols),
                                    LivePlot('If_tot', x, ax=axes[0]),
                                    LivePlot('I0', x, ax=axes[1])])
    yield from plan2
    """
    ept = numpy.array([])
    det = [sclr, xs]

    last_time_pt = time.time()
    ringbuf = collections.deque(maxlen=10)
    # c2pitch_kill=EpicsSignal("XF:05IDA-OP:1{Mono:HDCM-Ax:P2}Cmd:Kill-Cmd")
    xs.external_trig.put(False)

    # @bpp.stage_decorator([xs])
    yield from abs_set(xs.settings.acquire_time, 0.1)
    yield from abs_set(xs.total_points, 100)

    roi_name = "roi{:02}".format(roinum[0])
    roi_key = []
    roi_key.append(getattr(xs.channel1.rois, roi_name).value.name)
    livetableitem.append(roi_key[0])
    livecallbacks.append(LiveTable(livetableitem))
    liveploty = roi_key[0]
    liveplotx = energy.energy.name
    liveplotfig = plt.figure("raw xanes")
    livecallbacks.append(LivePlot(liveploty, x=liveplotx, fig=liveplotfig))

    myscan = count

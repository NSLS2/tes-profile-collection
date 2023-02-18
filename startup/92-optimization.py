# import matplotlib.pyplot as plt

try:
    from bloptools.de_opt_utils import run_hardware_fly
    from bloptools.de_optimization import optimization_plan
except (ImportError, ModuleNotFoundError):
    # The modules were moved to the `de` package in https://github.com/NSLS-II/bloptools/pull/5.
    from bloptools.de.de_opt_utils import run_hardware_fly
    from bloptools.de.de_optimization import optimization_plan

motor_dict = {sample_stage.x.name: {'position': sample_stage.x},
              sample_stage.y.name: {'position': sample_stage.y},
              # sample_stage.z.name: {'position': sample_stage.z},
              }

bound_vals = [(43, 44), (34, 35)]
motor_bounds = {}
motor_dict_keys = list(motor_dict.keys())
for k in range(len(motor_dict_keys)):
    motor_bounds[motor_dict_keys[k]] = {'position': [bound_vals[k][0],
                                                     bound_vals[k][1]]}

# instantiate plt.figure() before running optimization_plan
# plt.figure()

# Note: run it with
# RE(optimization_plan(fly_plan=run_hardware_fly, bounds=motor_bounds, db=db, motors=motor_dict,
# detector=xs, start_det=start_detector, read_det=read_detector, stop_det=stop_detector,
# watch_func=watch_function))

# Logbook: 2020-08-05

# Transient Scan ID: 2106     Time: 2020-08-05 14:33:11
# Persistent Unique Scan ID: '7744d3c6-7640-42e7-ae43-a404069cc181'
# !!! det reading: 110.0
# !!! det reading: 110.0
# !!! det reading: 110.0                                                                                                      | 0.0025/0.3955 [00:00<00:11, 28.91s/mm]
# !!! det reading: 66.0███████▏                                                                                            | 0.0284375/0.3955 [00:00<00:02,  7.97s/mm]
# !!! det reading: 71.0██████████████▏                                                                                     | 0.0559375/0.3955 [00:00<00:02,  6.81s/mm]
# !!! det reading: 37.0█████████████████████                                                                               | 0.0834375/0.3955 [00:00<00:02,  6.42s/mm]
# !!! det reading: 149.0███████████████████████████                                                                        | 0.1109375/0.3955 [00:00<00:01,  6.21s/mm]
# !!! det reading: 343.0██████████████████████████████████                                                                 | 0.1384375/0.3955 [00:00<00:01,  6.09s/mm]
# !!! det reading: 230.0█████████████████████████████████████████▉                                                           | 0.16625/0.3955 [00:00<00:01,  6.00s/mm]
# !!! det reading: 196.0████████████████████████████████████████████████▉                                                    | 0.19375/0.3955 [00:01<00:01,  5.94s/mm]
# !!! det reading: 89.0█████████████████████████████████████████████████████████                                             | 0.22125/0.3955 [00:01<00:01,  5.90s/mm]
# !!! det reading: 79.0██████████████████████████████████████████████████████████████████▋                                   | 0.25875/0.3955 [00:01<00:00,  5.85s/mm]
# !!! det reading: 201.0████████████████████████████████████████████████████████████████████████▊                            | 0.28625/0.3955 [00:01<00:00,  5.83s/mm]
# !!! det reading: 60.0████████████████████████████████████████████████████████████████████████████████▉                     | 0.31375/0.3955 [00:01<00:00,  5.81s/mm]
# !!! det reading: 124.0███████████████████████████████████████████████████████████████████████████████████████              | 0.34125/0.3955 [00:01<00:00,  5.79s/mm]
# !!! det reading: 156.0██████████████████████████████████████████████████████████████████████████████████████████████       | 0.36875/0.3955 [00:02<00:00,  5.77s/mm]
# !!! det reading: 205.0████████████████████████████████████████████████████████████████████████████████████████████████████▌| 0.39375/0.3955 [00:02<00:00,  5.80s/mm]
# !!! det reading: 205.025mm [00:02,  6.33s/mm]
# New stream: 'tes_hardware_flyer'

# In [7]: hdr = db['7744d3c6-7640-42e7-ae43-a404069cc181']
#
# In [8]: hdr.table(stream_name='tes_hardware_flyer', fields=['tes_hardware_flyer_sample_stage_x_position', 'tes_hardware_flyer_sample_stage_y_position', 'tes_hardwar
#    ...: e_flyer_intensity'])
# Out[8]:
#                                  time  tes_hardware_flyer_intensity  tes_hardware_flyer_sample_stage_x_position  tes_hardware_flyer_sample_stage_y_position
# seq_num
# 1       2020-08-05 14:33:14.213995934                         205.0                                   54.365938                                   40.369062
#
# In [9]:

# ^^^ solved by rewriting the watch function (no globals)

# TODO:
# 1) have a live plot for the fitness (convergence plot)
# 2) improve the algorithm to optimize the travel path (should not go back to the same point)
# 3) Usability:
#    - convenient I/O to the saved data (pandas dataframes, sorting, ...)
#    - clean up prints and better output log
#    - add metadata about the samples
#    - plot trajectory

# dofs = [kbv.ush, kbv.dsh]
# #dofs = [kbh.ush, kbh.dsh]
dofs = [toroidal_mirror.ush, toroidal_mirror.dsh]
dofs = [toroidal_mirror.usy, toroidal_mirror.dsy, toroidal_mirror.ush, toroidal_mirror.dsh]
dofs = [kbv.ush, kbv.dsh, kbh.ush, kbh.dsh]

#dofs = [kbv.ush, kbv.dsh, kbh.ush, kbh.dsh, toroidal_mirror.usy, toroidal_mirror.dsy, toroidal_mirror.ush, toroidal_mirror.dsh]

rel_bounds = {"kbv_ush": [-1e-1, +1e-1], "kbv_dsh": [-1e-1, +1e-1], "kbh_ush": [-1e-1, +1e-1], "kbh_dsh": [-1e-1, +1e-1], "toroidal_mirror_ush": [-1e-1, +1e-1], "toroidal_mirror_dsh": [-1e-1, +1e-1], "toroidal_mirror_usy": [-1e-1, +1e-1], "toroidal_mirror_dsy": [-1e-1, +1e-1]}
fid_params = {"kbv_ush": -0.0500010, "kbv_dsh": -0.0500010, "kbh_ush": 2.2650053, "kbh_dsh": 3.3120017, "toroidal_mirror_ush": -9.515, "toroidal_mirror_dsh": -3.92, "toroidal_mirror_usy": -6.284, "toroidal_mirror_dsy": -9.2575}
hard_bounds = np.r_[[fid_params[dof.name] + np.array(rel_bounds[dof.name]) for dof in dofs]].T

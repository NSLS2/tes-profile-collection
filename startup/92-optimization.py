print(f"Loading {__file__!r} ...")

import os
import sys
import textwrap
import traceback

from tabulate import tabulate


def _extract_tb():
    """Auxiliary function to provide a pretty-printed traceback of the exceptions.

    This is useful when used in the try..except blocks so that the printed
    exception traceback can be easily distinguished from the actual exceptions.

    Example:
    --------
    ╭──────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ Traceback                                                                                            │
    ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ File ".../.ipython/profile_collection_tes/startup/92-optimization.py", line 43, in <module>     from │
    │ bloptools.de_opt_utils import run_hardware_fly                                                       │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯
        ╭──────────────────────────────────────────╮
        │ Exception                                │
        ├──────────────────────────────────────────┤
        │ No module named 'bloptools.de_opt_utils' │
        ╰──────────────────────────────────────────╯

    """
    shared_kwargs = {"tablefmt": "rounded_grid"}
    traceback_kwargs = {**shared_kwargs}
    two_borders = 2 * 2
    indent = 4

    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = traceback.format_tb(exc_traceback)

    max_len = max([len(line) for line in tb])
    term_width = os.get_terminal_size().columns

    tb = "".join(tb)

    if max_len >= term_width - two_borders:
        traceback_kwargs.update({"maxcolwidths": term_width - two_borders})

    print(tabulate([[tb]], headers=("Traceback",), **traceback_kwargs))
    print(
        textwrap.indent(
            tabulate(
                [[exc_value]],
                headers=("Exception",),
                maxcolwidths=(term_width - two_borders - indent),
                **shared_kwargs,
            ),
            prefix=" " * indent,
        )
    )


if auto_alignment_mode():  # defined in 00-startup.py
    # Enable the auto-alignment mode.
    # In this mode, the KB-mirror motors an optionally, toroidal mirror motors
    # will be moved. For that to work, the I0 suspender will have to be disabled
    # to avoid the interference with the mono feedback system
    # (https://github.com/NSLS-II-TES/tes-horizontal-feedback).

    try:
        # Imports for bloptools < v0.0.2.
        from bloptools.de_opt_utils import run_hardware_fly
        from bloptools.de_optimization import optimization_plan
    except (ImportError, ModuleNotFoundError):
        _extract_tb()

        print("\nFalling back to a newer version of bloptools...\n")
        # The modules were moved to the `de` package in https://github.com/NSLS-II/bloptools/pull/5.
        # To be released as v0.1.0 or newer.
        try:
            from bloptools import gp
            from bloptools.de.de_opt_utils import run_hardware_fly
            from bloptools.de.de_optimization import optimization_plan
        except (ImportError, ModuleNotFoundError):
            _extract_tb()
            print(f"\nContinuing without bloptools...\n")

    ###########################################################################
    #                            DE optimization                              #
    ###########################################################################

    # motor_dict = {
    #     sample_stage.x.name: {"position": sample_stage.x},
    #     sample_stage.y.name: {"position": sample_stage.y},
    #     # sample_stage.z.name: {'position': sample_stage.z},
    # }

    # bound_vals = [(43, 44), (34, 35)]
    # motor_bounds = {}
    # motor_dict_keys = list(motor_dict.keys())
    # for k in range(len(motor_dict_keys)):
    #     motor_bounds[motor_dict_keys[k]] = {"position": [bound_vals[k][0], bound_vals[k][1]]}

    # instantiate plt.figure() before running optimization_plan
    # import matplotlib.pyplot as plt
    # plt.figure()

    # Usage:
    #
    # RE(optimization_plan(fly_plan=run_hardware_fly, bounds=motor_bounds, db=db,
    #                      motors=motor_dict, detector=xs, start_det=start_detector,
    #                      read_det=read_detector, stop_det=stop_detector,
    #                      watch_func=watch_function))

    ###########################################################################
    #                            GP optimization                              #
    ###########################################################################

    # dofs = [kbv.ush, kbv.dsh]
    # dofs = [kbh.ush, kbh.dsh]
    dofs = [toroidal_mirror.ush, toroidal_mirror.dsh]
    dofs = [
        toroidal_mirror.usy,
        toroidal_mirror.dsy,
        toroidal_mirror.ush,
        toroidal_mirror.dsh,
    ]
    dofs = np.array([kbv.ush, kbv.dsh, kbh.ush, kbh.dsh])
    # dofs = [kbv.ush, kbv.dsh, kbh.ush, kbh.dsh, toroidal_mirror.usy, toroidal_mirror.dsy, toroidal_mirror.ush, toroidal_mirror.dsh]

    rel_bounds = {
        "kbv_ush": [-1e-1, +1e-1],
        "kbv_dsh": [-1e-1, +1e-1],
        "kbh_ush": [-1e-1, +1e-1],
        "kbh_dsh": [-1e-1, +1e-1],
        "toroidal_mirror_ush": [-1e-1, +1e-1],
        "toroidal_mirror_dsh": [-1e-1, +1e-1],
        "toroidal_mirror_usy": [-1e-1, +1e-1],
        "toroidal_mirror_dsy": [-1e-1, +1e-1],
    }
    fid_params = {
        "kbv_ush": -0.0500010,
        "kbv_dsh": -0.0500010,
        "kbh_ush": 2.2650053,
        "kbh_dsh": 3.3120017,
        "toroidal_mirror_ush": -9.515,
        "toroidal_mirror_dsh": -3.92,
        "toroidal_mirror_usy": -6.284,
        "toroidal_mirror_dsy": -9.2575,
    }
    hard_bounds = np.r_[[fid_params[dof.name] + 2 * np.array(rel_bounds[dof.name]) for dof in dofs]]

    # gpo = gp.Optimizer(init_scheme='quasi-random', n_init=4, run_engine=RE, db=db, shutter=psh, detector=vstream, detector_type='image', dofs=dofs, dof_bounds=hard_bounds, fitness_model='max_sep_density', training_iter=256, verbose=True)

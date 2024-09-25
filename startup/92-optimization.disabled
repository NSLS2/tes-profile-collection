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


#if auto_alignment_mode():  # defined in 00-startup.py
    # Enable the auto-alignment mode.
    # In this mode, the KB-mirror motors an optionally, toroidal mirror motors
    # will be moved. For that to work, the I0 suspender will have to be disabled
    # to avoid the interference with the mono feedback system
    # (https://github.com/NSLS-II-TES/tes-horizontal-feedback).

    ###########################################################################
    #                            GP optimization                              #
    ###########################################################################

    # dofs = [kbv.ush, kbv.dsh]
    # dofs = [kbh.ush, kbh.dsh]

from blop import DOF, Objective, Agent


kb_radius = 0.1

toroid_radius = 0.02

kbh_ush_bounds = 2.33 + kb_radius * np.array([-1, +1])
kbh_dsh_bounds = 3.36 + kb_radius * np.array([-1, +1])

kbv_ush_bounds = .15 + kb_radius * np.array([-1, +1])
kbv_dsh_bounds = .13 + kb_radius * np.array([-1, +1])

toroid_ush_bounds = -9.51 + toroid_radius * np.array([-1, +1])
toroid_dsh_bounds = -3.858 + toroid_radius * np.array([-1, +1])

#kbh_ush_bounds = kbh.dsh.read()["kbh_ush"]["value"] + np.array([-0.05, 0.05])
#kbh_dsh_bounds = kbh.dsh.read()["kbh_dsh"]["value"] + np.array([-0.05, 0.05])


dofs = [
    DOF(device=kbh.ush, search_bounds=kbh_ush_bounds, description="KBH upstream", tags=["kb"], active=False),
    DOF(device=kbh.dsh, search_bounds=kbh_dsh_bounds, description="KBH downstream", tags=["kb"], active=False),
    DOF(device=kbv.ush, search_bounds=kbv_ush_bounds, description="KBV upstream", tags=["kb"], active=False),
    DOF(device=kbv.dsh, search_bounds=kbv_dsh_bounds, description="KBV downstream", tags=["kb"], active=False),
    DOF(device=toroidal_mirror.ush, search_bounds=toroid_ush_bounds, description="Toroid upstream height", tags=["toroid"]),
    DOF(device=toroidal_mirror.dsh, search_bounds=toroid_dsh_bounds, description="Toroid downstream height", tags=["toroid"]),
]

objs = [
    Objective(name="I0", target="max", log=True, latent_groups=[("toroidal_mirror_ush", "toroidal_mirror_dsh")]),
    Objective(name="wid_x", target="min", log=True, latent_groups=[("kbh_ush", "kbh_dsh")], weight=0),
    Objective(name="wid_y", target="min", log=True, latent_groups=[("kbv_ush", "kbv_dsh")], weight=0),
]

dets = [vstream, I0]

from blop.utils.misc import best_image_feedback

def digestion(db, uid):

    products = db[uid].table(fill=True)

    for index, entry in products.iterrows():

        im = entry.vstream_image

        ny, nx = im.shape

        x0, xw, y0, yw = best_image_feedback(im)

        bad = False
        bad |= x0 < 16
        bad |= x0 > nx - 16
        bad |= y0 < 16
        bad |= y0 > ny - 16

        if bad:
            x0, xw, y0, yw = 4 * [np.nan]

        products.loc[index, "pos_x"] = x0
        products.loc[index, "pos_y"] = y0
        products.loc[index, "wid_x"] = xw 
        products.loc[index, "wid_y"] = yw


    return products


agent = Agent(dofs=dofs, objectives=objs, dets=dets, digestion=digestion, db=db)





    # dofs = [toroidal_mirror.ush, toroidal_mirror.dsh]
    # dofs = [
    #     toroidal_mirror.usy,
    #     toroidal_mirror.dsy,
    #     toroidal_mirror.ush,
    #     toroidal_mirror.dsh,
    # ]
    # dofs = np.array([kbv.ush, kbv.dsh, kbh.ush, kbh.dsh])
    # # dofs = [kbv.ush, kbv.dsh, kbh.ush, kbh.dsh, toroidal_mirror.usy, toroidal_mirror.dsy, toroidal_mirror.ush, toroidal_mirror.dsh]

    # rel_bounds = {
    #     "kbv_ush": [-1e-1, +1e-1],
    #     "kbv_dsh": [-1e-1, +1e-1],
    #     "kbh_ush": [-1e-1, +1e-1],
    #     "kbh_dsh": [-1e-1, +1e-1],
    #     "toroidal_mirror_ush": [-1e-1, +1e-1],
    #     "toroidal_mirror_dsh": [-1e-1, +1e-1],
    #     "toroidal_mirror_usy": [-1e-1, +1e-1],
    #     "toroidal_mirror_dsy": [-1e-1, +1e-1],
    # }
    # fid_params = {
    #     "kbv_ush": -0.0500010,
    #     "kbv_dsh": -0.0500010,
    #     "kbh_ush": 2.2650053,
    #     "kbh_dsh": 3.3120017,
    #     "toroidal_mirror_ush": -9.515,
    #     "toroidal_mirror_dsh": -3.92,
    #     "toroidal_mirror_usy": -6.284,
    #     "toroidal_mirror_dsy": -9.2575,
    # }
    # hard_bounds = np.r_[[fid_params[dof.name] + 2 * np.array(rel_bounds[dof.name]) for dof in dofs]]

    # # gpo = gp.Optimizer(init_scheme='quasi-random', n_init=4, run_engine=RE, db=db, shutter=psh, detector=vstream, detector_type='image', dofs=dofs, dof_bounds=hard_bounds, fitness_model='max_sep_density', training_iter=256, verbose=True)

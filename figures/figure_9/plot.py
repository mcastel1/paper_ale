import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import constants.utils as const
import calculus.geometry as geo
import graphics.color_bar as cb
import graphics.mesh.utils as msh
import graphics.utils as gr
import graphics.vector_plot as vp
import input_output.utils as io
import list.column_labels as clab
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec


matplotlib.use('pgf')  # use a non-interactive backend to avoid the need of

# Show all rows and columns when printing a Pandas array
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))


# add the path where to find the shared modules
module_path = paths.root_path + "/figures/modules/"
sys.path.append(module_path)

# Suppress the specific warning
warnings.filterwarnings(
    "ignore", message=".*Z contains NaN values.*", category=UserWarning)
# clean the matplotlib cache to load the correct version of definitions.tex
os.system("rm -rf ~/.matplotlib/tex.cache")

pplt.rc['grid'] = False  # disables default gridlines

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": (
        r"\usepackage{newpxtext,newpxmath} "
        r"\usepackage{bm} "
        r"\usepackage{xcolor} "
        r"\usepackage{tikz} "
        r"\usetikzlibrary{math} "
        r"\usepackage{glossaries} "
        rf"\input{{{paths.definitions_path}}}"
        rf"\input{{{os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../definitions.tex')}}}"
    )
})

print("Current working directory:", os.getcwd())
print("root_path:", os.path.dirname(os.path.abspath(__file__)))


solution_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/")
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])
snapshot_path = os.path.join(solution_path, "snapshots/csv/")
snapshot_nodal_values_path = os.path.join(snapshot_path, "nodal_values")


# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max('line_mesh_n_', snapshot_path)
number_of_frames = snapshot_max - snapshot_min + 1


data_ref_boundary_vertices_sub_mesh_1 = pd.read_csv(os.path.join(
    mesh_path, 'boundary_points_id_' + str(parameters['sub_mesh_1_id']) + '.csv'))


fig = pplt.figure(
    figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
    left=parameters['figure_margin_l'],
    bottom=parameters['figure_margin_b'],
    right=parameters['figure_margin_r'],
    top=parameters['figure_margin_t'],
    wspace=parameters['wspace'],
    hspace=parameters['hspace'])

# pre-create subplots and axes
fig.add_subplot(2, 1, 1)
fig.add_subplot(2, 1, 2)


def plot_snapshot(fig, n_file,
                  axis_min_max=None):

    n_file_string = str(n_file)

    # load data
    data_msh_ref_line_vertices = pd.read_csv(os.path.join(
        mesh_path, 'line_vertices.csv'))
    data_msh_curr_line_vertices = pd.read_csv(os.path.join(
        snapshot_path, 'line_mesh_n_' + n_file_string + '.csv'))
    data_X = pd.read_csv(os.path.join(
        snapshot_path, 'X_n_12_' + n_file_string + '.csv'))

    if axis_min_max == None:

        # compute the min and max of the axes
        #
        data_u_msh = pd.read_csv(os.path.join(
            snapshot_nodal_values_path, 'u_n_' + str(n_file) + '.csv'))

        X_msh_ref, Y_msh_ref, u_msh_n_X, u_msh_n_Y, _, _, _, _ = vp.interpolate_2d_vector_field(data_u_msh,
                                                                                                [0, 0],
                                                                                                [parameters['L'],
                                                                                                    parameters['h']],
                                                                                                parameters['n_bins_v'])

        # X, Y are the positions of the mesh nodes in the current configuration
        X = np.array(lis.add_lists_of_lists(X_msh_ref, u_msh_n_X))
        Y = np.array(lis.add_lists_of_lists(Y_msh_ref, u_msh_n_Y))

        # compute the min-max of the snapshot
        axis_min_max = [lis.min_max(X), lis.min_max(Y)]
        #

    X_curr, _ = gr.interpolate_curve(
        data_X, axis_min_max[0][0], axis_min_max[0][1], parameters['n_bins_X'])

    X_msh_ref, Y_msh_ref, u_msh_n_X, u_msh_n_Y, _, _, _, _ = vec.interpolate_2d_vector_field(data_u_msh,
                                                                                             [0, 0],
                                                                                             [parameters['L'],
                                                                                                 parameters['h']],
                                                                                             parameters['n_bins_v'],
                                                                                             clab.label_x_column,
                                                                                             clab.label_y_column,
                                                                                             clab.label_v_column)

    # =============
    # reference subplot
    # =============

    ax = fig.axes[0]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,
                       [0, 0], [parameters['L'], parameters['h']]
                       )

    # compute the vector field u and store it in U_x, U_y and its related coordinates X_U, Y_U in the current configuration
    X_U, Y_U, _, _ = geo.u_1d(data_X, parameters['h'])
    # coordinates of the curve in the reference configuration
    X_ref = np.array(list(zip(X_U, Y_U)))

    # plot \partial Omega_in
    start = [0, 0]
    end = [0, parameters['h']]
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_in_color'],
        linewidth=parameters['X_line_width'],
        linestyle=':',
        label=r'$\pomineqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_out
    start = [parameters['L'], 0]
    end = [parameters['L'], parameters['h']]
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_out_color'],
        linewidth=parameters['X_line_width'],
        linestyle='--',
        label=r'$\pomouteqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_bottom
    start = [0, 0]
    end = [parameters['L'], 0]
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_bottom_color'],
        linewidth=parameters['X_line_width'],
        linestyle='-.',
        label='$\pombottomeqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_top
    start = [0, parameters['h']]
    end = [parameters['L'], parameters['h']]
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_top_color'],
        linewidth=parameters['X_line_width'],
        linestyle='-',
        label='$\pomtopeqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot partial_Omega_line_in
    ax.plot([0], [parameters['h']], 'o',
            s=parameters['partial_omega_line_point_size'],
            color=parameters['partial_omega_line_in_color'],
            linewidth=parameters['X_line_width'],
            label=r'$\pomlineineqr$',
            zorder=const.high_z_order,
            clip_on=False
            )

    # plot partial_Omega_line_out
    ax.plot(
        [parameters['L']], [parameters['h']],
        marker='o',
        markersize=parameters['partial_omega_line_point_size'],
        markerfacecolor='none',          # empty
        markeredgecolor=parameters['partial_omega_line_out_color'],
        markeredgewidth=parameters['X_line_width'],
        linestyle='None',
        label=r'$\pomlineouteqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot ref mesh
    gr.plot_2d_mesh(ax, data_msh_ref_line_vertices,
                    line_width=parameters['mesh_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])

    # Create custom legend handles
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=handles,
        labels=labels,
        loc='center',
        bbox_to_anchor=np.array(parameters['legend_position_ref']),
        frameon=True,
        handlelength=parameters['legend_line_length'],
        prop=FontProperties(size=parameters['legend_font_size'])
    )

    gr.plot_2d_axes(
        ax, [0, 0], [parameters['L'], parameters['h']],
        tick_length=parameters['tick_length'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['ref_axis_label'],
        axis_label_angle=parameters['axis_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters['ref_panel_label'],
        plot_label_offset=parameters['ref_panel_label_offset'],
        plot_label_font_size=parameters['panel_label_font_size'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order)

    # =============
    # current subplot
    # =============

    ax = fig.axes[1]

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)
    gr.set_axes_limits(ax,
                       [0, 0], [parameters['L'], parameters['h']]
                       )

    # compute the vector field u and store it in U_x, U_y and its related coordinates X_U, Y_U in the current configuration
    X_U, Y_U, _, _ = geo.u_1d(data_X, parameters['h'])

    # store the interpolating field for the displacement into U_interp
    U_interp_x, U_interp_y = vp.interpolating_function_2d_vector_field(
        data_u_msh)
    U_interp = np.array([U_interp_x, U_interp_y], dtype=object)

    # coordinates of the curve in the reference configuration
    # X_ref = np.array(list(zip(X_U, Y_U)))

    # plot curr mesh
    gr.plot_2d_mesh(ax, data_msh_curr_line_vertices,
                    line_width=parameters['mesh_line_width'],
                    color='black',
                    alpha=parameters['alpha_mesh'],
                    zorder=parameters['mesh_zorder'])

    # plot X_curr
    gr.plot_curve_grid(ax, X_curr,
                       line_color=parameters['partial_omega_top_color'],
                       legend='\pomtopeqc',
                       line_width=parameters['X_line_width'],
                       z_order=1
                       )

    # plot \partial Omega_in
    start = msh.reference_to_current([0, 0], U_interp)
    end = msh.reference_to_current(
        [0, parameters['h']], U_interp)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_in_color'],
        linewidth=parameters['X_line_width'],
        linestyle=':',
        label=r'$\pomineqc$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_out
    start = msh.reference_to_current([parameters['L'], 0], U_interp)
    end = msh.reference_to_current(
        [parameters['L'], parameters['h']], U_interp)

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_out_color'],
        linewidth=parameters['X_line_width'],
        linestyle='--',
        label=r'$\pomouteqc$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_bottom
    start = msh.reference_to_current([0, 0], U_interp)
    end = msh.reference_to_current(
        [parameters['L'], 0], U_interp)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=parameters['partial_omega_bottom_color'],
        linewidth=parameters['X_line_width'],
        linestyle='-.',
        label='$\pombottomeqc$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot partial_Omega_line_in
    point = msh.reference_to_current(
        [0, parameters['h']], U_interp)
    ax.plot([point[0]], [point[1]], 'o',
            s=parameters['partial_omega_line_point_size'],
            color=parameters['partial_omega_line_in_color'],
            linewidth=parameters['X_line_width'],
            label=r'$\pomlineineqc$',
            zorder=const.high_z_order,
            clip_on=False
            )

    # plot partial_Omega_line_out
    point = msh.reference_to_current(
        [parameters['L'], parameters['h']], U_interp)
    ax.plot([point[0]], [point[1]], 'o',
            s=parameters['partial_omega_line_point_size'],
            color=parameters['partial_omega_line_out_color'],
            linewidth=parameters['X_line_width'],
            label=r'$\pomlineouteqc$',
            markerfacecolor='none',
            linestyle='None',
            zorder=const.high_z_order,
            clip_on=False
            )

    # Create custom legend handles
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles=handles,
        labels=labels,
        loc='center',
        bbox_to_anchor=np.array(parameters['legend_position_cur']),
        frameon=True,
        handlelength=parameters['legend_line_length'],
        prop=FontProperties(size=parameters['legend_font_size'])
    )

    gr.plot_2d_axes(
        ax, [0, 0], [parameters['L'], parameters['h']],
        tick_length=parameters['tick_length'],
        line_width=parameters['axis_line_width'],
        axis_label=parameters['cur_axis_label'],
        axis_label_angle=parameters['axis_label_angle'],
        axis_label_offset=parameters['axis_label_offset'],
        tick_label_offset=parameters['tick_label_offset'],
        tick_label_format=['f', 'f'],
        font_size=parameters['axis_font_size'],
        plot_label=parameters['curr_panel_label'],
        plot_label_offset=parameters['cur_panel_label_offset'],
        plot_label_font_size=parameters['panel_label_font_size'],
        axis_origin=parameters['axis_origin'],
        margin=parameters['axis_margin'],
        n_minor_ticks=parameters['n_minor_ticks'],
        minor_tick_length=parameters['minor_tick_length'],
        z_order=const.high_z_order)


plot_snapshot(fig, snapshot_max)

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

# pplt.show()

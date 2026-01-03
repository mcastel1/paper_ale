import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import proplot as pplt
import sys
import warnings

import calculus.utils as cal
import calculus.geometry as geo
import constants.utils as const
import graphics.utils as gr
import list.column_labels as clab
import input_output.utils as io
import list.utils as lis
import system.paths as paths
import system.utils as sys_utils
import graphics.vector_plot as vec

# use this to show tikz objects in the plot
matplotlib.use('pgf')


parameters = io.read_parameters_from_csv_file(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'parameters.csv'))


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

# define the folder where to read the data
solution_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "solution/")
mesh_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/")
sub_mesh_1_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "mesh/solution/sub_meshes/out/")
snapshot_path = os.path.join(solution_path, 'snapshots/csv/')
figure_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), parameters['figure_name'])


# compute the min and max snapshot present in the solution path
snapshot_min, snapshot_max = sys_utils.n_min_max(
    'line_mesh_msh_n_', snapshot_path)

number_of_frames = snapshot_max - snapshot_min + 1


data_boundary_vertices_ellipse = pd.read_csv(os.path.join(
    mesh_path, 'boundary_points_id_' + str(parameters['ellipse_loop_id']) + '.csv'))

fig = pplt.figure(figsize=(parameters['figure_size'][0], parameters['figure_size'][1]),
                  left=parameters['figure_margin_l'],
                  bottom=parameters['figure_margin_b'],
                  right=parameters['figure_margin_r'],
                  top=parameters['figure_margin_t'],
                  wspace=parameters['wspace'],
                  hspace=parameters['hspace'])


# pre-create subplots and axes
fig.add_subplot(2, 1, 1)
fig.add_subplot(2, 1, 2)


def plot_snapshot(fig, n_file):

    n_snapshot = str(n_file)

    data_u_msh_ref = pd.read_csv(
        solution_path + 'snapshots/csv/nodal_values/u_msh_n_' + str(1) + '.csv')
    data_u_msh_cur = pd.read_csv(
        solution_path + 'snapshots/csv/nodal_values/u_msh_n_' + n_snapshot + '.csv')

    # =============
    # reference configuration subplot
    # =============

    ax = fig.axes[0]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # load data for the first snapshot (reference configuration)
    data_el_line_vertices = pd.read_csv(
        solution_path + 'snapshots/csv/line_mesh_el_n_' + str(1) + '.csv')
    data_msh_line_vertices = pd.read_csv(
        solution_path + 'snapshots/csv/line_mesh_msh_n_' + str(1) + '.csv')

    # plot the polygon of the boundary 'ellipse_loop_id'
    #
    # build two a vector field which interpolates the displacement field in data_u_msh
    U_interp_x, U_interp_y = vec.interpolating_function_2d_vector_field(
        data_u_msh_ref)

    # run through points in data_boundary_vertices_ellipse (reference configuration) and add to them [U_interp_x, U_interp_y] in order to obtain the boundary polygon in the current configuration
    data_def_boundary_vertices_ellipse = []
    for _, row in data_boundary_vertices_ellipse.iterrows():
        data_def_boundary_vertices_ellipse.append(
            np.add(
                [row[':0'], row[':1']],
                [U_interp_x(row[':0'], row[':1']),
                 U_interp_y(row[':0'], row[':1'])]
            )
        )

    # plot \partial Omega_in
    ax.plot(
        [0, 0],
        [0, parameters['h']],
        color=parameters['partial_omega_in_color'],
        linewidth=parameters['partial_omega_line_width'],
        linestyle='-.',
        label='$\pomineqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_out
    ax.plot(
        [parameters['L'], parameters['L']],
        [0, parameters['h']],
        color=parameters['partial_omega_out_color'],
        linewidth=parameters['partial_omega_line_width'],
        linestyle=':',
        label='$\pomouteqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_top
    ax.plot(
        [0, parameters['L']],
        [parameters['h'], parameters['h']],
        color=parameters['partial_omega_top_color'],
        linewidth=parameters['partial_omega_line_width'],
        linestyle='--',
        label='$\pomtopeqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_bottom
    ax.plot(
        [0, parameters['L']],
        [0, 0],
        color=parameters['partial_omega_bottom_color'],
        linewidth=parameters['partial_omega_line_width'],
        dashes=[5, 2, 2, 2, 2, 2],
        label='$\pombottomeqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot the boundary polygon in the current configuration
    partial_omega_circle_out_ref = Polygon(data_def_boundary_vertices_ellipse, fill=False,
                                           linewidth=parameters['partial_omega_line_width'],
                                           edgecolor=parameters['partial_omega_circle_out_color'],
                                           zorder=const.high_z_order)

    ax.add_patch(partial_omega_circle_out_ref)

    # this is a dummy line used only to show the legend for polygon
    dummy_line_handle = Line2D([0], [0],
                               color=parameters['partial_omega_circle_out_color'],
                               linewidth=parameters['partial_omega_line_width'],
                               linestyle='-')

    # Create custom legend handles
    handles, labels = ax.get_legend_handles_labels()

    handles.append(dummy_line_handle)
    labels.append(r'$\pomcircouteqr$')

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices,
                    parameters['mesh_el_line_width'], 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    parameters['mesh_msh_line_width'], 'black', parameters['alpha_mesh'])

    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper right',
        bbox_to_anchor=np.array(parameters['legend_position']),
        frameon=True,
        handlelength=parameters['legend_line_length']
    )

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],
                    axis_label=parameters['axis_label_reference'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    tick_label_angle=parameters['tick_label_angle'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    axis_origin=parameters['axis_origin'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters["reference_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length'],
                    )

    # =============
    # current configuration subplot
    # =============

    ax = fig.axes[1]  # Use the existing axis

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.grid(False)  # <-- disables ProPlot's auto-enabled grid

    # load data for a snapshot > 1 -> current configuration
    data_el_line_vertices = pd.read_csv(
        solution_path + 'snapshots/csv/line_mesh_el_n_' + str(n_snapshot) + '.csv')
    data_msh_line_vertices = pd.read_csv(
        solution_path + 'snapshots/csv/line_mesh_msh_n_' + str(n_snapshot) + '.csv')

    # plot the polygon of the boundary 'ellipse_loop_id'
    #
    # build two a vector field which interpolates the displacement field in data_u_msh
    U_interp_x, U_interp_y = vec.interpolating_function_2d_vector_field(
        data_u_msh_cur)

    # run through points in data_boundary_vertices_ellipse (reference configuration) and add to them [U_interp_x, U_interp_y] in order to obtain the boundary polygon in the current configuration
    data_def_boundary_vertices_ellipse = []
    for _, row in data_boundary_vertices_ellipse.iterrows():
        data_def_boundary_vertices_ellipse.append(
            np.add(
                [row[':0'], row[':1']],
                [U_interp_x(row[':0'], row[':1']),
                 U_interp_y(row[':0'], row[':1'])]
            )
        )

    # plot \partial Omega_in
    ax.plot(
        [0, 0],
        [0, parameters['h']],
        color=parameters['partial_omega_in_color'],
        linewidth=parameters['partial_omega_line_width'],
        linestyle='-.',
        label='$\pomineqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_out
    ax.plot(
        [parameters['L'], parameters['L']],
        [0, parameters['h']],
        color=parameters['partial_omega_out_color'],
        linewidth=parameters['partial_omega_line_width'],
        linestyle=':',
        label='$\pomouteqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_top
    ax.plot(
        [0, parameters['L']],
        [parameters['h'], parameters['h']],
        color=parameters['partial_omega_top_color'],
        linewidth=parameters['partial_omega_line_width'],
        linestyle='--',
        label='$\pomtopeqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot \partial Omega_bottom
    ax.plot(
        [0, parameters['L']],
        [0, 0],
        color=parameters['partial_omega_bottom_color'],
        linewidth=parameters['partial_omega_line_width'],
        dashes=[5, 2, 2, 2, 2, 2],
        label='$\pombottomeqr$',
        zorder=const.high_z_order,
        clip_on=False
    )

    # plot the boundary polygon in the current configuration
    partial_omega_circle_out_cur = Polygon(data_def_boundary_vertices_ellipse, fill=False,
                                           linewidth=parameters['partial_omega_line_width'],
                                           edgecolor=parameters['partial_omega_circle_out_color'],
                                           zorder=const.high_z_order)

    ax.add_patch(partial_omega_circle_out_cur)

    # this is a dummy line used only to show the legend for polygon
    dummy_line_handle = Line2D([0], [0],
                               color=parameters['partial_omega_circle_out_color'],
                               linewidth=parameters['partial_omega_line_width'],
                               linestyle='-')

    # Create custom legend handles
    handles, labels = ax.get_legend_handles_labels()

    handles.append(dummy_line_handle)
    labels.append(r'$\pomcircouteqc$')

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices,
                    parameters['mesh_el_line_width'], 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    parameters['mesh_msh_line_width'], 'black', parameters['alpha_mesh'])

    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper right',
        bbox_to_anchor=np.array(parameters['legend_position']),
        frameon=True,
        handlelength=parameters['legend_line_length']
    )

    gr.plot_2d_axes(ax, [0, 0], [parameters['L'], parameters['h']],
                    axis_label=parameters['axis_label_current'],
                    axis_label_angle=parameters['axis_label_angle'],
                    axis_label_offset=parameters['axis_label_offset'],
                    tick_label_offset=parameters['tick_label_offset'],
                    tick_label_format=parameters['tick_label_format'],
                    tick_label_angle=parameters['tick_label_angle'],
                    font_size=parameters['font_size'],
                    line_width=parameters['axis_line_width'],
                    axis_origin=parameters['axis_origin'],
                    tick_length=parameters['tick_length'],
                    plot_label=parameters["current_plot_panel_label"],
                    plot_label_offset=parameters['panel_label_position'],
                    plot_label_font_size=parameters['panel_label_font_size'],
                    n_minor_ticks=parameters['n_minor_ticks'],
                    minor_tick_length=parameters['minor_tick_length']
                    )


plot_snapshot(fig, snapshot_max)

# keep this also for the animation: it allows for setting the right dimensions to the animation frame
plt.savefig(figure_path + '_large.pdf')
os.system(
    f'magick -density {parameters["compression_density"]} {figure_path}_large.pdf -quality {parameters["compression_quality"]} -compress JPEG {figure_path}.pdf')

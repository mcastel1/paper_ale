import matplotlib
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

matplotlib.use('Agg')  # use a non-interactive backend to avoid the need of


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

    # gr.set_2d_axes_limits(ax, [0, 0], [parameters['L'], parameters['h']], [0, 0])

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices,
                    parameters['mesh_el_line_width'], 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    parameters['mesh_msh_line_width'], 'black', parameters['alpha_mesh'])

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

    # gr.set_2d_axes_limits(ax, [0, 0], [parameters['L'], parameters['h']], [0, 0])

    # plot mesh for elastic problem and for mesh oustide the elastic body
    gr.plot_2d_mesh(ax, data_el_line_vertices,
                    parameters['mesh_el_line_width'], 'red', parameters['alpha_mesh'])
    gr.plot_2d_mesh(ax, data_msh_line_vertices,
                    parameters['mesh_msh_line_width'], 'black', parameters['alpha_mesh'])

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
